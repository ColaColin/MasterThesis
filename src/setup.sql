-- run this to setup or reset the database to have all the tables needed

drop table if exists run_iteration_stats;
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
    creation timestamptz default NOW() not null,
    run uuid references runs (id) not null,
    acc_rnd_limited float,
    acc_best_limited float,
    acc_rnd_full float,
    acc_best_full float
);

--alter table networks drop column acc_rnd;
--alter table networks drop column acc_best;
--alter table networks add column acc_rnd_limited float;
--alter table networks add column acc_best_limited float;
--alter table networks add column acc_rnd_full float;
--alter table networks add column acc_best_full float;

create table states (
    id uuid primary key,
    package_size integer not null,
    worker varchar not null,
    creation timestamptz default NOW() not null,
    iteration integer not null,
    -- will be null in iteration 0, as there is no trained network for iteration 0. The random network is not submitted
    network uuid references networks(id),
    run uuid references runs (id) not null
);

create table run_iteration_stats (
    run uuid references runs (id) not null,
    iteration integer,

    played_states integer not null,
    new_states integer not null,
    
    first_player_wins float not null,
    draws float not null,

    game_length_avg float not null,
    game_length_std float not null,

    primary key (run, iteration)
);

create view runs_info as
    select r.id, r.name, r.creation, count(n.id) as iterations
    from runs r
    left join networks n on n.run = r.id
    group by r.id, r.name;