create table messages
(
    message_id numeric       not null primary key,
    guild_id   numeric       not null,
    user_id    numeric       not null,
    channel_id numeric       not null,
    content    varchar(5000) not null,
    time       timestamp     not null
);

create index messages_user_id_index
    on messages (user_id);

create index messages_guild_id_index
    on messages (guild_id);

create index messages_content_index
    on messages (content);

create index messages_time_index
    on messages (time);

create table models
(
    id              serial    not null,
    last_message_id numeric   not null primary key,
    guild_id        numeric   not null,
    user_id         numeric   not null,
    model_topology  text      not null,
    weight_specs    text      not null,
    weight_data     bytea     not null,
    tokenized_data  text      not null,
    update_time     timestamp not null
);

create index models_user_id_index
    on models (user_id);

create index models_guild_id_index
    on models (guild_id);

create index models_model_time_index
    on models (update_time);

create index models_last_message_id_index
    on models (last_message_id);