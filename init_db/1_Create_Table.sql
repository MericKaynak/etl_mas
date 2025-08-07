CREATE TABLE Student (
    ID INT PRIMARY KEY,
    Name VARCHAR(100),
    Email VARCHAR(100),
    Department VARCHAR(100),
    Password VARCHAR(20)
);

CREATE TABLE Teacher (
    ID INT PRIMARY KEY,
    Name VARCHAR(100),
    Email VARCHAR(100),
    Role VARCHAR(50),
    Weekly_Work_Hours INT
);

CREATE TABLE Room (
    ID INT PRIMARY KEY,
    Label VARCHAR(100),
    Capacity INT,
    Location VARCHAR(100)
);

CREATE TABLE Appointment (
    ID INT PRIMARY KEY,
    Weekday VARCHAR(100),
    Start_Time TIME,
    End_Time TIME,
    CONSTRAINT Unique_Appointment UNIQUE (Weekday, Start_Time, End_Time)
);

CREATE TABLE Course (
    ID INT PRIMARY KEY,
    Title VARCHAR(100),
    Department VARCHAR(100),
    Duration INT,
    Room_ID INT,
    Appointment_ID INT,
    Teacher_ID INT,
    FOREIGN KEY (Room_ID) REFERENCES Room(ID),
    FOREIGN KEY (Appointment_ID) REFERENCES Appointment(ID),
    FOREIGN KEY (Teacher_ID) REFERENCES Teacher(ID)
);

CREATE SEQUENCE attendance_seq START WITH 1 INCREMENT BY 1;

CREATE TABLE Attendance (
    ID INT PRIMARY KEY DEFAULT NEXTVAL('attendance_seq'),
    Student_ID INT,
    Course_ID INT,
    FOREIGN KEY (Student_ID) REFERENCES Student(ID),
    FOREIGN KEY (Course_ID) REFERENCES Course(ID)
);

CREATE TABLE Notification (
    ID INT PRIMARY KEY,
    Notification_Type VARCHAR(50),
    Text TEXT,
    Recipient_ID INT,
    Appointment_ID INT,
    FOREIGN KEY (Recipient_ID) REFERENCES Student(ID),
    FOREIGN KEY (Appointment_ID) REFERENCES Appointment(ID)
);

CREATE TABLE Administrator (
    ID INT PRIMARY KEY,
    Name VARCHAR(100),
    Email VARCHAR(100),
    Password VARCHAR(20)
);

CREATE SEQUENCE substitution_seq START WITH 1 INCREMENT BY 1;

CREATE TABLE Substitution (
    ID INT PRIMARY KEY DEFAULT NEXTVAL('substitution_seq'),
    Course_ID INT,
    Date DATE,
    Teacher_ID INT,
    FOREIGN KEY (Course_ID) REFERENCES Course(ID),
    FOREIGN KEY (Teacher_ID) REFERENCES Teacher(ID)
);

CREATE TABLE Curriculum_Schedule (
    ID INT PRIMARY KEY,
    Course_ID INT,
    Date DATE,
    Substitution_ID INT DEFAULT NULL,
    FOREIGN KEY (Substitution_ID) REFERENCES Substitution(ID),
    FOREIGN KEY (Course_ID) REFERENCES Course(ID)
);

-- Domain knowledge table to describe all tables and their purposes

CREATE TABLE domain_knowledge (
    Table_Name VARCHAR(100) PRIMARY KEY,
    Description TEXT
);