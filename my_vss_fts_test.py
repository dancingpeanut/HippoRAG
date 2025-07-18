import sqlite3
import sqlite_vss


db = sqlite3.connect(':memory:')
db.enable_load_extension(True)
sqlite_vss.load(db)

def test_vss():
    pass


if __name__ == '__main__':
    version, = db.execute('select vss_version()').fetchone()
    print(version)

    test_vss()
