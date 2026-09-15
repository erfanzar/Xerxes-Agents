// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import {expect,test} from 'bun:test'
import {parseDelivery} from '../src/desktop/renderer/Deliveries.js'
test('delivery record preserves destination, attempt revision and saved content',()=>{expect(parseDelivery({id:'one',platform:'slack',recipient:'room',state:'uncertain',attempts:2,content:'saved\noutput'})).toMatchObject({recipient:'room',attempts:2,content:'saved\noutput',state:'uncertain'})})
test('invalid delivery state or attempt cannot enable actions',()=>{for(const value of [{state:'made-up',attempts:1},{state:'pending',attempts:-1},{state:'pending',attempts:1.2}])expect(()=>parseDelivery({id:'one',platform:'slack',recipient:'room',...value})).toThrow()})
