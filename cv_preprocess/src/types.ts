/** Shared image containers for cv_preprocess. */

export interface RGBAImage {
  width: number;
  height: number;
  data: Uint8Array; // w*h*4, RGBA8
}

export interface GrayImage {
  width: number;
  height: number;
  data: Uint8Array; // w*h, single channel 0..255
}
