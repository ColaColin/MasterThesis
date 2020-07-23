from core.mains.command import getCommandConfiguration

import psycopg2
from psycopg2 import pool

if __name__ == "__main__":
    config = getCommandConfiguration()

    pool = psycopg2.pool.SimpleConnectionPool(1, 20,user = config["dbuser"],
                                            password = config["dbpassword"],
                                            host = "127.0.0.1",
                                            port = "5432",
                                            database = config["dbname"]);


    try:
        con = pool.getconn()
        cursor = con.cursor()

        cursor.execute("SELECT name, id, sha, config from runs")

        rows = cursor.fetchall()

        for row in rows:
            rname = row[0]
            id = row[1]
            sha = row[2]
            config = row[3]

            fcontent = "# Run Name: " + rname + "\n"
            fcontent += "# Run ID:  " + id + "\n"
            fcontent += "# Run SHA: " + sha + "\n"
            fcontent += config

            with open("../configs/runs/" + id + ".yaml", "w") as f:
                f.write(fcontent)

    finally:
        pool.putconn(con)
    


    