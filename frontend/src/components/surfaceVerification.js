export function rawToDisplay(point, transform) {
  const x = transform.horizontalFlip ? transform.imageWidth - point.x : point.x;
  return { x: x * transform.scale + transform.offsetX, y: point.y * transform.scale + transform.offsetY };
}

export function displayToRaw(point, transform) {
  const displayX = (point.x - transform.offsetX) / transform.scale;
  return { x: transform.horizontalFlip ? transform.imageWidth - displayX : displayX,
    y: (point.y - transform.offsetY) / transform.scale };
}

export function swapSurfaceRecords(surfaces) {
  return surfaces.map(surface => ({ ...surface, surface: surface.surface === 'mesial' ? 'distal' : 'mesial' }));
}

export function markerX(bbox, pixelSide) {
  return pixelSide === 'image_left' ? bbox[0] : bbox[2];
}
