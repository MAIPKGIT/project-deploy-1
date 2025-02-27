import os
import hashlib

class Config:
    # ถ้าไม่มีตัวแปรใน environment จะใช้คีย์ที่สุ่มขึ้น
    SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(24).hex())  # หรือใช้ hashlib ได้เช่นกัน
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', hashlib.sha256(os.urandom(24)).hexdigest())
    JWT_ACCESS_TOKEN_EXPIRES = 86400  # 1 day
    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root@localhost/kmitl_project"
    # SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:First1058@database-kitchen:3306/kmitl_project"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
