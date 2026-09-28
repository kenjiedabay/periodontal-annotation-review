export type Rect={left:number;top:number;width:number;height:number};
export function clientToOriginal(clientX:number,clientY:number,rect:Rect,imageWidth:number,imageHeight:number):{x:number;y:number};
export function cropClientToOriginal(clientX:number,clientY:number,rect:Rect,crop:number[]):{x:number;y:number};
export function inBounds(point:{x:number;y:number},width:number,height:number):boolean;
export function editState(raw:Record<string,unknown>,operation:'add'|'move'|'reassign'|'delete'|'cancel',payload?:Record<string,any>):any;
