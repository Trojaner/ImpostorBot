import {Sequelize, DataTypes} from 'sequelize';

const sequelize = new Sequelize(process.env.CONNECTION_STRING as string);
export const DbMessages = sequelize.define(
  'messages',
  {
    message_id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
    },
    guild_id: DataTypes.INTEGER,
    user_id: DataTypes.INTEGER,
    channel_id: DataTypes.INTEGER,
    content: DataTypes.STRING,
    time: DataTypes.DATE,
  },
  {
    timestamps: false,
  }
);

export const DbModels = sequelize.define(
  'models',
  {
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
    },
    last_message_id: DataTypes.INTEGER,
    guild_id: DataTypes.INTEGER,
    user_id: DataTypes.INTEGER,
    model_topology: DataTypes.TEXT,
    weight_specs: DataTypes.TEXT,
    weight_data: DataTypes.BLOB,
    tokenized_data: DataTypes.TEXT,
    update_time: DataTypes.DATE,
  },
  {
    timestamps: false,
  }
);
