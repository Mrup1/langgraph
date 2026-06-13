import sqlite3
import os

DB_NAME = "hospital.db"

def init_db():
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 1. Doctors Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Doctors (
            doctor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            specialty TEXT NOT NULL,
            experience_years INTEGER,
            contact_info TEXT
        )
    ''')

    # 2. Services Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Services (
            service_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER,
            service_name TEXT NOT NULL,
            price REAL,
            duration_minutes INTEGER,
            FOREIGN KEY (doctor_id) REFERENCES Doctors(doctor_id)
        )
    ''')

    # 3. Schedules Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Schedules (
            schedule_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER,
            day_of_week TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            FOREIGN KEY (doctor_id) REFERENCES Doctors(doctor_id)
        )
    ''')

    # 4. Appointments Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Appointments (
            appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER,
            patient_name TEXT NOT NULL,
            appointment_date TEXT NOT NULL,
            appointment_time TEXT NOT NULL,
            status TEXT DEFAULT 'Booked',
            FOREIGN KEY (doctor_id) REFERENCES Doctors(doctor_id)
        )
    ''')

    # 5. HospitalInfo Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS HospitalInfo (
            info_key TEXT PRIMARY KEY,
            info_value TEXT NOT NULL
        )
    ''')

    # --- Populating Sizable Dummy Data ---
    
    doctors_data = [
        ("Dr. Arsalan Khan", "Cardiology", 14, "dr.arsalan@hospital.com"),
        ("Dr. Sarah Ahmed", "Cardiology", 8, "dr.sarah@hospital.com"),
        ("Dr. Zainab Bilal", "Pediatrics", 10, "dr.zainab@hospital.com"),
        ("Dr. Bilal Cheema", "Neurology", 12, "dr.bilal@hospital.com"),
        ("Dr. Ayesha Malik", "Dermatology", 6, "dr.ayesha@hospital.com")
    ]
    cursor.executemany("INSERT INTO Doctors (name, specialty, experience_years, contact_info) VALUES (?, ?, ?, ?)", doctors_data)

    services_data = [
        (1, "Cardiology Consultation", 150.0, 30),
        (1, "Echocardiogram", 300.0, 45),
        (2, "Cardiology Consultation", 120.0, 30),
        (3, "Pediatric Checkup", 100.0, 20),
        (3, "Vaccination Routine", 50.0, 15),
        (4, "Neurological Evaluation", 200.0, 40),
        (4, "EEG Tracking Study", 450.0, 60),
        (5, "Skin Consultation", 90.0, 20),
        (5, "Laser Treatment Therapy", 500.0, 45)
    ]
    cursor.executemany("INSERT INTO Services (doctor_id, service_name, price, duration_minutes) VALUES (?, ?, ?, ?)", services_data)

    schedules_data = [
        # Dr. Arsalan Khan (Cardiology) - Mon, Wed
        (1, "Monday", "09:00", "13:00"),
        (1, "Wednesday", "09:00", "13:00"),
        # Dr. Sarah Ahmed (Cardiology) - Tue, Thu
        (2, "Tuesday", "14:00", "18:00"),
        (2, "Thursday", "14:00", "18:00"),
        # Dr. Zainab Bilal (Pediatrics) - Mon, Tue, Wed, Thu, Fri
        (3, "Monday", "10:00", "15:00"),
        (3, "Wednesday", "10:00", "15:00"),
        # Dr. Bilal Cheema (Neurology) - Fri
        (4, "Friday", "09:00", "16:00"),
        # Dr. Ayesha Malik (Dermatology) - Sat
        (5, "Saturday", "11:00", "16:00")
    ]
    cursor.executemany("INSERT INTO Schedules (doctor_id, day_of_week, start_time, end_time) VALUES (?, ?, ?, ?)", schedules_data)

    appointments_data = [
        (1, "John Doe", "2026-06-15", "09:30"),
        (1, "Alice Smith", "2026-06-15", "10:00"),
        (3, "Baby Emma", "2026-06-15", "11:00"),
        (4, "Robert Downey", "2026-06-19", "14:00")
    ]
    cursor.executemany("INSERT INTO Appointments (doctor_id, patient_name, appointment_date, appointment_time) VALUES (?, ?, ?, ?)", appointments_data)

    info_data = [
        ("location", "The hospital is located at Sector Phase 5, Hayatabad, Peshawar, near the main commercial hub."),
        ("emergency", "Our Emergency Ward runs 24/7. It features critical trauma units, cardiac support setups, and on-call surgeons."),
        ("visiting_hours", "General visiting hours are daily from 02:00 PM to 05:00 PM and 07:00 PM to 09:00 PM."),
        ("insurance_policy", "We accept all corporate healthcare policies along with major private insurances including Sehat Card Plus."),
        ("pharmacy", "The in-house hospital pharmacy is located on the ground floor next to the OPD and operates 24 hours a day.")
    ]
    cursor.executemany("INSERT INTO HospitalInfo (info_key, info_value) VALUES (?, ?)", info_data)

    conn.commit()
    conn.close()
    print("Database initialization completed successfully.")

if __name__ == "__main__":
    init_db()