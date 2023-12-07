// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

module {
  AIE.device(ipu) {
    %0 = AIE.tile(0, 0)
    %1 = AIE.tile(0, 1)
    %2 = AIE.tile(0, 2)
    %rtp = AIE.buffer(%2) {sym_name = "rtp"} : memref<16xi32>
    AIE.objectFifo @objFifo_in0(%0, {%1}, 2 : i32) : !AIE.objectFifo<memref<640xi32>>
    AIE.objectFifo @objFifo_in1(%1, {%2}, 2 : i32) : !AIE.objectFifo<memref<640xi32>>
    AIE.objectFifo.link [@objFifo_in0] -> [@objFifo_in1] ()
    AIE.objectFifo @objFifo_out0(%1, {%0}, 2 : i32) : !AIE.objectFifo<memref<640xi32>>
    AIE.objectFifo @objFifo_out1(%2, {%1}, 2 : i32) : !AIE.objectFifo<memref<640xi32>>
    AIE.objectFifo.link [@objFifo_out1] -> [@objFifo_out0] ()
    func.func private @threshold4ChLine(%in: memref<640xi32>, %out: memref<640xi32>, %lineWidth: i32,  %thresholdValue1: i16, %thresholdValue2: i16, %thresholdValue3: i16, %thresholdValue4: i16, %maxValue1: i16, %maxValue2: i16, %maxValue3: i16, %maxValue4: i16, %thresholdType: i8) -> ()
    %21 = AIE.core(%2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c_maxint = arith.constant 0xFFFFFFFF : index
      %lineWidth = arith.constant 2560 : i32
      %maxValue = arith.constant 255 : i16
      scf.for %arg0 = %c0 to %c_maxint step %c1 {
        %subview0 = AIE.objectFifo.acquire @objFifo_in1(Consume, 1) : !AIE.objectFifoSubview<memref<640xi32>>
        %elem0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<640xi32>> -> memref<640xi32>
        %subview1 = AIE.objectFifo.acquire @objFifo_out1(Produce, 1) : !AIE.objectFifoSubview<memref<640xi32>>
        %elem1 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<640xi32>> -> memref<640xi32>
        %thresh1         = memref.load %rtp[%c0] : memref<16xi32>
        %thresholdValue1 = arith.trunci %thresh1 : i32 to i16
        %thresh2         = memref.load %rtp[%c1] : memref<16xi32>
        %thresholdValue2 = arith.trunci %thresh2 : i32 to i16
        %thresh3         = memref.load %rtp[%c2] : memref<16xi32>
        %thresholdValue3 = arith.trunci %thresh3 : i32 to i16
        %thresh4         = memref.load %rtp[%c3] : memref<16xi32>
        %thresholdValue4 = arith.trunci %thresh4 : i32 to i16
        %tt = memref.load %rtp[%c4] : memref<16xi32>
        %threshType = arith.trunci %tt : i32 to i8
        func.call @threshold4ChLine(%elem0, %elem1, %lineWidth, %thresholdValue1, %thresholdValue2, %thresholdValue3, %thresholdValue4, %maxValue, %maxValue, %maxValue, %maxValue, %threshType) : (memref<640xi32>, memref<640xi32>, i32, i16, i16, i16, i16, i16, i16, i16, i16, i8) -> ()
        AIE.objectFifo.release @objFifo_in1(Consume, 1)
        AIE.objectFifo.release @objFifo_out1(Produce, 1)
      }
      AIE.end
    } {link_with = "threshold.o"}
    // For 720p rgba:
    func.func @sequence(%in : memref<921600xi32>, %out : memref<921600xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c921600 = arith.constant 921600 : i32
      AIEX.ipu.rtp_write(0, 2, 0, 3) { buffer_sym_name = "rtp" }  // thresholdValue1
      AIEX.ipu.rtp_write(0, 2, 1, 4) { buffer_sym_name = "rtp" }  // thresholdValue2
      AIEX.ipu.rtp_write(0, 2, 4, 7) { buffer_sym_name = "rtp" }  // thresholdType
      AIEX.ipu.rtp_write(0, 2, 2, 5) { buffer_sym_name = "rtp" }  // thresholdValue3
      AIEX.ipu.rtp_write(0, 2, 3, 6) { buffer_sym_name = "rtp" }  // thresholdValue4
      AIEX.ipu.dma_memcpy_nd (%c0, %c0, %out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c921600][%c0,%c0,%c0]) { metadata = @objFifo_out0, id = 1 : i32 } : (i32, i32, memref<921600xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
      AIEX.ipu.dma_memcpy_nd (%c0, %c0, %in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c921600][%c0,%c0,%c0]) { metadata = @objFifo_in0, id = 0 : i32 } : (i32, i32, memref<921600xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
      AIEX.ipu.sync { column = 0 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
      return
    }
  }
}

