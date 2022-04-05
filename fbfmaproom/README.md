# FbF Maproom

## Adding a foreign table

This maproom makes use of foreign tables in Postgres. Here's a brief explanation of how to add one:

1. Login in the pgdb12 server using psql. Make sure the user/role you use has the appropriate permissions

    psql -h pgdb12.iri.columbia.edu -U fist -W DesignEngine

2. Create a server. Note that this has probably already been done so should be unnecessary

    CREATE SERVER dlcomputemon1_iridb FOREIGN DATA WRAPPER postgres_fdw OPTIONS (host 'dlcomputemon1.iri.columbia.edu', port '5432', dbname 'iridb');

3. Import the specific table(s) you want. It is advised to use the command below and not `IMPORT FOREIGN TABLE` so that the
   definition of the table does not have to be (re)specified by hand. The schema can be arbitrarily chosen, I've used public here
   but it might be advisable to create a schema for foreign tables specifically. Make sure you have the right schema of the table in the foreign server.

    IMPORT FOREIGN SCHEMA public LIMIT TO (table_name) FROM SERVER dlcomputemon1_iridb INTO public;

4. Create a user mapping for every user in pgdb12 you want to have access to tables in the foreign server

    CREATE USER MAPPING FOR dero SERVER dlcomputemon1_iridb OPTIONS (user 'ingrid_ro', password 'PASSWORD');

5. Grant select privileges to the local user if necessary

    GRANT SELECT ON table_name TO dero;
