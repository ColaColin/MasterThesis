-- run this to setup or reset the database to have all the tables needed

drop view if exists runs_info;
drop table if exists states;
drop table if exists networks;
drop table if exists runs;

create table runs (
    id uuid primary key,
    name varchar not null,
    config varchar not null,
    creation timestamptz default NOW() not null
);

create table networks (
    id uuid primary key,
    fpath varchar not null,
    creation timestamptz default NOW() not null,
    iteration integer not null,
    run uuid references runs (id) not null
);

create table states (
    id uuid primary key,
    -- can be calculated from the id as binary/id[0]/id[1]/id[2]/id
    -- fpath varchar not null,
    package_size integer not null,
    worker varchar not null,
    creation timestamptz default NOW() not null,
    iteration integer not null,
    -- will be null in iteration 0, as there is no trained network for iteration 0. The random network is not submitted
    network uuid references networks(id),
    run uuid references runs (id) not null
);

create view runs_info as
    select r.id, r.name, r.creation, count(n.id) as iterations
    from runs r
    left join networks n on n.run = r.id
    group by r.id, r.name;