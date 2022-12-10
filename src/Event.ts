import {ClientEvents} from 'discord.js';

import ImpersonatorClient from './ImpersonatorClient';

export class Event {
  name!: keyof ClientEvents;

  run!: (client: ImpersonatorClient, ...eventArgs: any) => any;

  constructor(options: NonNullable<Event>) {
    Object.assign(this, options);
  }
}
