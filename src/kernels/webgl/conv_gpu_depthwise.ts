/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Conv2DInfo} from '../../ops/conv_util';
import {GPGPUProgram} from './gpgpu_math';

export class DepthwiseConv2DProgram implements GPGPUProgram {
  variableNames = ['x', 'W'];
  outputShape: number[];
  userCode: string;

  constructor(convInfo: Conv2DInfo) {
    this.outputShape = convInfo.outShape;

    const xNumRows = convInfo.inHeight;
    const xNumCols = convInfo.inWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const channelMul = convInfo.outChannels / convInfo.inChannels;

    const filterSize = filterHeight * filterWidth;
    const nearestVec4 = Math.floor(filterSize / 4) * 4;
    const vec4Remainder = filterSize % 4;

    this.userCode = `

      float sampleX(int w, int batch, int xRCorner, int xCCorner, int d1) {
        int wR = w / ${filterWidth};
        int xR = xRCorner + wR * ${dilationHeight};
        if (xR < 0 || xR >= ${xNumRows}) {
          return 0.0;
        }
        int wC = w - wR * ${filterWidth};
        int xC = xCCorner + wC * ${dilationWidth};
        if (xC < 0 || xC >= ${xNumCols}) {
          return 0.0;
        }
        return getX(batch, xR, xC, d1);
      }

      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${channelMul};
        int q = d2 - d1 * ${channelMul};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int w = 0; w < ${nearestVec4}; w += 4) {
          vec4 xValues = vec4(
            sampleX(w, batch, xRCorner, xCCorner, d1),
            sampleX(w + 1, batch, xRCorner, xCCorner, d1),
            sampleX(w + 2, batch, xRCorner, xCCorner, d1),
            sampleX(w + 3, batch, xRCorner, xCCorner, d1)
          );
          vec4 wValues = vec4(
            getW(w, d1, q),
            getW(w + 1, d1, q),
            getW(w + 2, d1, q),
            getW(w + 3, d1, q)
          );
          dotProd += dot(xValues, wValues);
        }
        if (${vec4Remainder === 1}) {
          dotProd += sampleX(${nearestVec4}, batch, xRCorner, xCCorner, d1) *
              getW(${nearestVec4}, d1, q);
        } else if (${vec4Remainder === 2}) {
          vec2 xValues = vec2(
            sampleX(${nearestVec4}, batch, xRCorner, xCCorner, d1),
            sampleX(${nearestVec4} + 1, batch, xRCorner, xCCorner, d1)
          );
          vec2 wValues = vec2(
            getW(${nearestVec4}, d1, q),
            getW(${nearestVec4} + 1, d1, q)
          );
          dotProd += dot(xValues, wValues);
        } else if (${vec4Remainder === 3}) {
          vec3 xValues = vec3(
            sampleX(${nearestVec4}, batch, xRCorner, xCCorner, d1),
            sampleX(${nearestVec4} + 1, batch, xRCorner, xCCorner, d1),
            sampleX(${nearestVec4} + 2, batch, xRCorner, xCCorner, d1)
          );
          vec3 wValues = vec3(
            getW(${nearestVec4}, d1, q),
            getW(${nearestVec4} + 1, d1, q),
            getW(${nearestVec4} + 2, d1, q)
          );
          dotProd += dot(xValues, wValues);
        }

        setOutput(dotProd);
      }
    `;
  }
}
