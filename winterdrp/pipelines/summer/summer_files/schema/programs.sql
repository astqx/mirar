CREATE TABLE IF NOT EXISTS programs (
    progid SERIAL PRIMARY KEY,
    progname VARCHAR(20),
    programid INT,
    subprogname VARCHAR(20),
    piname VARCHAR(20),
    mjd_start REAL,
    mjd_end REAL,
    time_hours REAL,
    progpri REAL
);