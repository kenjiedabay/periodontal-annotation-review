export type PilotStep={id:string;label:string};
export const PILOT_STEPS:PilotStep[];
export function tutorialKey(expertId:string):string;
export function tutorialComplete(storage:Storage,expertId:string):boolean;
export function finishTutorial(storage:Storage,expertId:string):void;
export function stepComplete(step:string,orientation:Record<string,unknown>,teeth:unknown[]):boolean;
export function nextIncomplete(states:unknown[]):number;
export function caseLabel(state:unknown):string;
