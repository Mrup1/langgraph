import sqlite3
from datetime import datetime, timedelta
from langchain_core.tools import tool
from typing import Optional

DB_NAME = "hospital.db"


# =========================================================
# 1. FETCH DOCTORS (SAFE + OPTIONAL FILTER)
# =========================================================
@tool
def fetch_doctors(specialty: Optional[str] = None) -> str:
    """
    Fetch hospital doctors.

    Args:
        specialty: Optional filter for doctor specialization

    Returns:
        List of doctors with details
    """

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    if specialty:
        cursor.execute(
            "SELECT name, specialty, experience_years, contact_info FROM Doctors WHERE specialty LIKE ?",
            (f"%{specialty}%",)
        )
    else:
        cursor.execute(
            "SELECT name, specialty, experience_years, contact_info FROM Doctors"
        )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No doctors found."

    return "\n".join(
        [f"Name: {r[0]} | Specialty: {r[1]} | Experience: {r[2]} yrs | Contact: {r[3]}"
         for r in rows]
    )


# =========================================================
# 2. FETCH PRICING (SAFE FILTERS)
# =========================================================
@tool
def fetch_pricing(
    doctor_name: Optional[str] = None,
    service_name: Optional[str] = None
) -> str:
    """
    Fetch consultation and service pricing.

    Args:
        doctor_name: optional doctor filter
        service_name: optional service filter

    Returns:
        Pricing details for hospital services
    """

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    query = """
        SELECT d.name, s.service_name, s.price, s.duration_minutes 
        FROM Services s 
        JOIN Doctors d ON s.doctor_id = d.doctor_id
        WHERE 1=1
    """
    params = []

    if doctor_name:
        query += " AND d.name LIKE ?"
        params.append(f"%{doctor_name}%")

    if service_name:
        query += " AND s.service_name LIKE ?"
        params.append(f"%{service_name}%")

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No pricing records found."

    return "\n".join(
        [f"Doctor: {r[0]} | Service: {r[1]} | Price: ${r[2]} | Duration: {r[3]} mins"
         for r in rows]
    )


# =========================================================
# 3. CHECK AVAILABILITY (FULLY SAFE)
# =========================================================
@tool
def check_availability(
    doctor_name: Optional[str] = None,
    date: Optional[str] = None
) -> str:
    """
    Check doctor availability and free slots.

    Args:
        doctor_name: doctor name (optional)
        date: YYYY-MM-DD (optional, defaults to today)

    Returns:
        Available time slots or guidance message
    """

    # DEFAULT DATE
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    # VALIDATE DATE
    try:
        day_of_week = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD."

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # If no doctor provided → list doctors instead of crashing
    if not doctor_name:
        cursor.execute("SELECT name, specialty FROM Doctors")
        doctors = cursor.fetchall()
        conn.close()

        return (
            "Please specify a doctor name.\n\nAvailable doctors:\n" +
            "\n".join([f"- {d[0]} ({d[1]})" for d in doctors])
        )

    # GET SCHEDULE
    cursor.execute("""
        SELECT d.doctor_id, s.start_time, s.end_time 
        FROM Doctors d
        JOIN Schedules s ON d.doctor_id = s.doctor_id
        WHERE d.name LIKE ? AND s.day_of_week = ?
    """, (f"%{doctor_name}%", day_of_week))

    schedule = cursor.fetchone()

    if not schedule:
        conn.close()
        return f"No schedule found for {doctor_name} on {date} ({day_of_week})."

    doc_id, start_str, end_str = schedule

    # BOOKED SLOTS
    cursor.execute("""
        SELECT appointment_time 
        FROM Appointments 
        WHERE doctor_id = ? 
        AND appointment_date = ? 
        AND status = 'Booked'
    """, (doc_id, date))

    booked_slots = [r[0] for r in cursor.fetchall()]
    conn.close()

    # GENERATE TIME SLOTS
    start_time = datetime.strptime(start_str, "%H:%M")
    end_time = datetime.strptime(end_str, "%H:%M")

    available_slots = []
    current = start_time

    while current < end_time:
        slot = current.strftime("%H:%M")
        if slot not in booked_slots:
            available_slots.append(slot)
        current += timedelta(minutes=30)

    if not available_slots:
        return f"No available slots for {doctor_name} on {date}."

    return f"Available slots for {doctor_name} on {date}:\n" + ", ".join(available_slots)


# =========================================================
# 4. BOOK APPOINTMENT (SAFE + VALIDATED)
# =========================================================
@tool
def book_appointment(
    doctor_name: str,
    patient_name: str,
    date: Optional[str] = None,
    time: Optional[str] = None
) -> str:
    """
    Book a hospital appointment.

    Args:
        doctor_name: doctor name
        patient_name: patient name
        date: YYYY-MM-DD
        time: HH:MM

    Returns:
        Booking confirmation or error message
    """

    if not date or not time:
        return "Please provide both date (YYYY-MM-DD) and time (HH:MM)."

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT doctor_id FROM Doctors WHERE name LIKE ?",
        (f"%{doctor_name}%",)
    )
    res = cursor.fetchone()

    if not res:
        conn.close()
        return f"Doctor '{doctor_name}' not found."

    doc_id = res[0]

    try:
        cursor.execute("""
            INSERT INTO Appointments 
            (doctor_id, patient_name, appointment_date, appointment_time, status)
            VALUES (?, ?, ?, ?, 'Booked')
        """, (doc_id, patient_name, date, time))

        conn.commit()
        conn.close()

        return f"Appointment booked for {patient_name} with {doctor_name} on {date} at {time}."

    except Exception as e:
        conn.close()
        return f"Booking failed: {str(e)}"


# =========================================================
# 5. GENERAL INFO TOOL (SAFE FAQ)
# =========================================================
@tool
def get_general_info(query: str = "") -> str:
    """
    Get hospital general information and FAQs.

    Args:
        query: user question (optional)

    Returns:
        Matching FAQ or full list
    """

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT info_key, info_value FROM HospitalInfo")
    rows = cursor.fetchall()
    conn.close()

    if not query:
        return "\n".join([f"- {r[0]}: {r[1]}" for r in rows])

    q = query.lower()

    for key, value in rows:
        if key.replace("_", " ") in q:
            return f"{key}: {value}"

    return "No match found.\n\n" + "\n".join([f"- {r[0]}: {r[1]}" for r in rows])


# =========================================================
# 6. MASTER TOOL (FULL OVERVIEW)
# =========================================================
@tool
def get_all_doctor_details() -> str:
    """
    Get complete list of all doctors with specialties and experience.

    Returns:
        Full doctor directory
    """

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name, specialty, experience_years, contact_info 
        FROM Doctors
    """)

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No doctors found."

    return "\n".join(
        [f"{r[0]} | {r[1]} | {r[2]} yrs | {r[3]}" for r in rows]
    )


# =========================================================
# EXPORT ALL TOOLS
# =========================================================
hospital_tools = [
    get_all_doctor_details,
    fetch_doctors,
    fetch_pricing,
    check_availability,
    book_appointment,
    get_general_info
]