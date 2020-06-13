#! /bin/bash


# pull a full backup of all data from the server to my local system.


echo "Creating database dump on the server"
ssh -t root@x0.cclausen.eu "rm /root/x0_backup.sql; PGPASSWORD=x0 PGHOST=127.0.0.1 PGDATABASE=x0 PGPORT=5432 PGUSER=x0 pg_dump --clean -w > /root/x0_backup.sql"

echo "Moving database dump to local machine"
scp root@x0.cclausen.eu:/root/x0_backup.sql /ImbaKeks/x0_backup.sql

echo "Restoring database dump on local machine"
PGPASSWORD=x0 PGHOST=127.0.0.1 PGDATABASE=x0 PGPORT=5432 PGUSER=x0 psql < /ImbaKeks/x0_backup.sql

echo "Copy binary data from the server to the local machine"
# rsync is completely overloaded with the number of files invovled
# rsync -r root@x0.cclausen.eu:/root/x0/ /ImbaKeks/x0_backup/

mkdir /ImbaKeks/x0_backup/
# tar files and transfer them as a single stream. This needs more bandwidth, but starts to transfer files directly instead of preparing for hours.
ssh root@x0.cclausen.eu "tar zcfv - x0/" | tar zxf - -C /ImbaKeks/x0_backup/

