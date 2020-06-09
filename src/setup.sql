-- run this to setup or reset the database to have all the tables needed

drop table if exists run_iteration_stats cascade;
drop view if exists runs_info cascade;
drop table if exists states cascade;
drop table if exists networks cascade;
drop table if exists runs cascade;
drop table if exists league_players cascade;
drop table if exists league_matches cascade;
drop table if exists league_players_snapshot cascade;

create table runs (
    id uuid primary key,
    name varchar not null,
    config varchar not null,
    sha varchar not null default 'unknown',
    creation timestamptz default NOW() not null
);

--alter table runs add column sha varchar not null default 'unknown';

create table networks (
    id uuid primary key,
    creation timestamptz default NOW() not null,
    run uuid references runs (id) on delete cascade not null,
    acc_network_moves float,
    acc_network_wins float,
    acc_mcts_moves float,
    frametime float
);

--alter table networks drop column acc_rnd_limited;
--alter table networks drop column acc_best_limited;
--alter table networks drop column acc_rnd_full;
--alter table networks drop column acc_best_full;
--alter table networks add column acc_network_moves float;
--alter table networks add column acc_network_wins float;
--alter table networks add column acc_mcts_moves float;

--alter table networks add column frametime float;

create table states (
    id uuid primary key,
    package_size integer not null,
    worker varchar not null,
    creation timestamptz default NOW() not null,
    iteration integer not null,
    -- will be null in iteration 0, as there is no trained network for iteration 0. The random network is not submitted
    network uuid references networks(id),
    run uuid references runs (id)  on delete cascade not null
);

create table run_iteration_stats (
    run uuid references runs (id) on delete cascade not null,
    iteration integer,

    played_states integer not null,
    new_states integer not null,
    
    first_player_wins float not null,
    draws float not null,

    game_length_avg float not null,
    game_length_std float not null,

    avg_nodes float not null default 0,

    primary key (run, iteration)
);

create table league_players (
    id uuid primary key,
    run uuid references runs (id) on delete cascade not null,
    rating float not null,
    -- stringified json for simplicity sake, the parameter data is very generic
    parameter_vals varchar not null,
    parameter_stddevs varchar not null,
    generation integer not null default 1
);

create table league_players_snapshot (
    id uuid references league_players (id) on delete cascade not null,
    generation integer not null,
    network uuid references networks(id) on delete cascade not null,
    rating float not null,
    primary key (id, generation)
);

create table league_matches(
    id serial primary key,
    run uuid references runs (id) on delete cascade not null,
    network uuid references networks(id) on delete cascade,
    player1 uuid references league_players (id) on delete cascade not null,
    player2 uuid references league_players (id) on delete cascade not null,
    result float not null,
    ratingChange float not null,
    creation timestamptz not null

);

create view runs_info as
    select r.id, r.name, r.creation, count(n.id) as iterations
    from runs r
    left join networks n on n.run = r.id
    group by r.id, r.name;