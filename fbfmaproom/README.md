# FbF Maproom

## Adding a foreign table

This maproom makes use of foreign tables in Postgres. Here's a brief explanation of how to add one:

1. Login in the pgdb12 server using psql. Make sure the user/role you use has the appropriate permissions

    psql -h pgdb12.iri.columbia.edu -U USER_NAME -W DesignEngine

2. Create a server. Note that this has probably already been done so should be unnecessary

    CREATE SERVER server_name FOREIGN DATA WRAPPER postgres_fdw OPTIONS (host 'HOST_NAME', port '5432', dbname 'DB_NAME');

3. Import the specific table(s) you want. It is advised to use the command below and not `IMPORT FOREIGN TABLE` so that the
   schema for the table does not have to be specified by hand. The schema can be arbitrarily chosen, I've used public here
   but it might be advisable to create a schema for foreign tables specifically. Make sure you have the right schema of the table in the foreign server.

    IMPORT FOREIGN SCHEMA public LIMIT TO (table_name) FROM SERVER server_name INTO public;

4. Create a user mapping for every user in pgdb12 you want to have access to tables in the foreign server

    CREATE USER MAPPING FOR local_user SERVER server_name OPTIONS (user 'FOREIGN_USER', password 'PASSWORD');

5. Grant select privileges to the local user if necessary

    GRANT SELECT ON table_name TO local_user;
