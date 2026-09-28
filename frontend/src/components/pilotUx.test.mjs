import assert from 'node:assert/strict';import test from 'node:test';
import{caseLabel,finishTutorial,nextIncomplete,stepComplete,tutorialComplete}from'./pilotUx.js';
test('tutorial first display finish and reopen preference',()=>{const data=new Map(),storage={getItem:k=>data.get(k)||null,setItem:(k,v)=>data.set(k,v)};assert.equal(tutorialComplete(storage,'e1'),false);finishTutorial(storage,'e1');assert.equal(tutorialComplete(storage,'e1'),true)});
test('required steps block continue',()=>{assert.equal(stepComplete(0,{status:'unverified'},[]),false);assert.equal(stepComplete(0,{status:'verified'},[]),true);const tooth={expert_fdi:'36',tooth_verification_status:'confirmed',surfaces:[]};assert.equal(stepComplete(1,{},[tooth]),true)});
test('dashboard and next incomplete do not use metrics',()=>{assert.equal(caseLabel({phase_a_locked:true}),'Phase A completed');assert.equal(nextIncomplete([{phase_a_locked:true,phase_b:{}},{phase_a_locked:false}]),1)});
