export type Vec3 = [number, number, number];
export type Mat3x3 = [Vec3, Vec3, Vec3];

export type GSCamera = {
  id: number;
  img_name: string;
  width: number;
  height: number;
  position: Vec3;
  rotation: Mat3x3;
  fx: number;
  fy: number;
};
