export type Point = { x: number; y: number };
export type Transform = { scale: number; offsetX: number; offsetY: number; imageWidth: number; horizontalFlip: boolean };
export function rawToDisplay(point: Point, transform: Transform): Point;
export function displayToRaw(point: Point, transform: Transform): Point;
export function swapSurfaceRecords<T extends { surface: 'mesial' | 'distal' }>(surfaces: T[]): T[];
export function markerX(bbox: number[], pixelSide: 'image_left' | 'image_right'): number;
