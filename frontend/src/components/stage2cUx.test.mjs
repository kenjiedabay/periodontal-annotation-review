import test from 'node:test';import assert from 'node:assert/strict';
import{caseStatus,findingName,nextUnreviewed}from'./stage2cUx.js';
const fs=[{source_annotation_id:'a',landmark_type:'cej'},{source_polyline_id:'b'}];
test('plain-language finding labels',()=>{assert.equal(findingName(fs[0]),'CEJ landmark');assert.equal(findingName(fs[1]),'Bone line')});
test('next finding skips reviewed records',()=>assert.equal(nextUnreviewed(fs,[{source_annotation_id:'a'}],0),1));
test('case status reflects append-only reviews',()=>{assert.equal(caseStatus(fs,[]),'Not started');assert.equal(caseStatus(fs,[{source_annotation_id:'a'}]),'In progress');assert.equal(caseStatus(fs,[{source_annotation_id:'a'},{source_annotation_id:'b'}]),'Complete')});
