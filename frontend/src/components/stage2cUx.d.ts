export function findingId(f:Record<string,unknown>):string;
export function findingName(f:Record<string,unknown>):string;
export function reviewedIds(actions:Array<Record<string,unknown>>):Set<string>;
export function nextUnreviewed(findings:Array<Record<string,unknown>>,actions:Array<Record<string,unknown>>,after?:number):number;
export function caseStatus(findings:Array<Record<string,unknown>>,actions:Array<Record<string,unknown>>):string;
