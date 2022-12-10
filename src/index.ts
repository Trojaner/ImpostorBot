#!/usr/bin/env node

import {config} from 'dotenv';

import ImpersonatorClient from './ImpersonatorClient';

config();

const client = new ImpersonatorClient();
client.start();
