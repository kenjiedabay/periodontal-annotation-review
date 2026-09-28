export function clientToOriginal(clientX,clientY,rect,imageWidth,imageHeight){
 return {x:(clientX-rect.left)*imageWidth/rect.width,y:(clientY-rect.top)*imageHeight/rect.height};
}
export function cropClientToOriginal(clientX,clientY,rect,crop){
 return {x:crop[0]+(clientX-rect.left)*(crop[2]-crop[0])/rect.width,y:crop[1]+(clientY-rect.top)*(crop[3]-crop[1])/rect.height};
}
export function inBounds(point,width,height){return point.x>=0&&point.y>=0&&point.x<width&&point.y<height}
export function editState(raw,operation,payload={}){
 const current={raw:{...raw},corrected:null,deleted:false,toothId:raw.toothId};
 if(operation==='add')return {...current,corrected:{...payload.point},toothId:payload.toothId};
 if(operation==='move')return {...current,corrected:{...payload.point}};
 if(operation==='reassign')return {...current,toothId:payload.toothId};
 if(operation==='delete')return {...current,deleted:true};
 return current;
}
