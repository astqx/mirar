CREATE TABLE IF NOT EXISTS raw (
    expid BIGINT PRIMARY KEY,
    savepath VARCHAR(255),
    obsdate INT,
    obsID INT,
    itid INT,
    night INT,
    fieldID INT,
    filter VARCHAR(5),
    progID INT,
    AExpTime FLOAT,
    expMJD FLOAT,
    subprog VARCHAR(20),
    airmass FLOAT,
    shutopen REAL,
    shutclsd REAL,
    tempture REAL,
    windspd REAL,
    Dewpoint REAL,
    Humidity REAL,
    Pressure REAL,
    Moonra REAL,
    Moondec REAL,
    Moonillf REAL,
    Moonphas REAL,
    Moonalt REAL,
    Sunaz REAL,
    Sunalt REAL,
    Detsoft VARCHAR(50),
    Detfirm VARCHAR(50),
    ra REAL,
    dec REAL,
    altitude REAL,
    azimuth REAL,
    procflag INT,
    rawcount SERIAL
);

CREATE INDEX ON raw (q3c_ang2pix(ra, dec));
CLUSTER raw_q3c_ang2pix_idx ON raw;
ANALYZE raw;

CREATE CLUSTERED INDEX raw_obsdate_idx ON raw (obsdate DESC);