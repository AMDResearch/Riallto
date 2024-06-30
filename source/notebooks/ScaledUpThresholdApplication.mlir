module  {
   AIE.device(ipu){

   %tile00 = AIE.tile(0, 0)
   %tile01 = AIE.tile(0, 1)
   %tile02 = AIE.tile(0, 2)
   %rtp_0_2 = AIE.buffer(%tile02) { sym_name = "rtp_0_2" } : memref<6xi32>
   %tile03 = AIE.tile(0, 3)
   %rtp_0_3 = AIE.buffer(%tile03) { sym_name = "rtp_0_3" } : memref<6xi32>
   %tile04 = AIE.tile(0, 4)
   %rtp_0_4 = AIE.buffer(%tile04) { sym_name = "rtp_0_4" } : memref<6xi32>
   %tile05 = AIE.tile(0, 5)
   %rtp_0_5 = AIE.buffer(%tile05) { sym_name = "rtp_0_5" } : memref<6xi32>
   AIE.objectFifo @itbuffer_0___ITout___mtbuffer_0___MTin(%tile00, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<1280xi32>>
   AIE.objectFifo @mtbuffer_0___MTout___rgba_rtp_thresh_0___in_buffer(%tile01, {%tile02}, 2 : i32) : !AIE.objectFifo<memref<320xi32>>
   AIE.objectFifo @mtbuffer_0___MTout___rgba_rtp_thresh_1___in_buffer(%tile01, {%tile03}, 2 : i32) : !AIE.objectFifo<memref<320xi32>>
   AIE.objectFifo @mtbuffer_0___MTout___rgba_rtp_thresh_2___in_buffer(%tile01, {%tile04}, 2 : i32) : !AIE.objectFifo<memref<320xi32>>
   AIE.objectFifo @mtbuffer_0___MTout___rgba_rtp_thresh_3___in_buffer(%tile01, {%tile05}, 2 : i32) : !AIE.objectFifo<memref<320xi32>>
   AIE.objectFifo @rgba_rtp_thresh_0___out_buffer___mtbuffer_1___MTin(%tile02, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<320xi32>>
   AIE.objectFifo @rgba_rtp_thresh_1___out_buffer___mtbuffer_1___MTin(%tile03, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<320xi32>>
   AIE.objectFifo @rgba_rtp_thresh_2___out_buffer___mtbuffer_1___MTin(%tile04, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<320xi32>>
   AIE.objectFifo @rgba_rtp_thresh_3___out_buffer___mtbuffer_1___MTin(%tile05, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<320xi32>>
   AIE.objectFifo @mtbuffer_1___MTout___itbuffer_1___ITin(%tile01, {%tile00}, 2 : i32) : !AIE.objectFifo<memref<1280xi32>>

   AIE.objectFifo.link [@itbuffer_0___ITout___mtbuffer_0___MTin ] -> [@mtbuffer_0___MTout___rgba_rtp_thresh_0___in_buffer,@mtbuffer_0___MTout___rgba_rtp_thresh_1___in_buffer,@mtbuffer_0___MTout___rgba_rtp_thresh_2___in_buffer,@mtbuffer_0___MTout___rgba_rtp_thresh_3___in_buffer] ()
   AIE.objectFifo.link [@rgba_rtp_thresh_0___out_buffer___mtbuffer_1___MTin,@rgba_rtp_thresh_1___out_buffer___mtbuffer_1___MTin,@rgba_rtp_thresh_2___out_buffer___mtbuffer_1___MTin,@rgba_rtp_thresh_3___out_buffer___mtbuffer_1___MTin ] -> [@mtbuffer_1___MTout___itbuffer_1___ITin] ()

   func.func private @rgba_rtp_thresh(memref<320xi32>, memref<320xi32>, i32, i32, i32, i32) -> ()

   AIE.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      %c_rtpidx_2 = arith.constant 2 : index
      %c_rtpidx_3 = arith.constant 3 : index
      %c_rtpidx_4 = arith.constant 4 : index
      %c_rtpidx_5 = arith.constant 5 : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview2 = AIE.objectFifo.acquire @mtbuffer_0___MTout___rgba_rtp_thresh_0___in_buffer(Consume, 1) : !AIE.objectFifoSubview<memref<320xi32>>
         %elem2 = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<320xi32>> -> memref<320xi32>
         %subview6 = AIE.objectFifo.acquire @rgba_rtp_thresh_0___out_buffer___mtbuffer_1___MTin(Produce, 1) : !AIE.objectFifoSubview<memref<320xi32>>
         %elem6 = AIE.objectFifo.subview.access %subview6[0] : !AIE.objectFifoSubview<memref<320xi32>> -> memref<320xi32>

         %nbytes = memref.load %rtp_0_2[%c_rtpidx_2] : memref<6xi32>
         %r_thresh = memref.load %rtp_0_2[%c_rtpidx_3] : memref<6xi32>
         %g_thresh = memref.load %rtp_0_2[%c_rtpidx_4] : memref<6xi32>
         %b_thresh = memref.load %rtp_0_2[%c_rtpidx_5] : memref<6xi32>
         func.call @rgba_rtp_thresh(%elem2, %elem6, %nbytes, %r_thresh, %g_thresh, %b_thresh) : (memref<320xi32>, memref<320xi32>, i32, i32, i32, i32) -> ()

         AIE.objectFifo.release @mtbuffer_0___MTout___rgba_rtp_thresh_0___in_buffer(Consume, 1)
         AIE.objectFifo.release @rgba_rtp_thresh_0___out_buffer___mtbuffer_1___MTin(Produce, 1)
      }
      AIE.end
   } { link_with="rgba_rtp_thresh.o" }

   AIE.core(%tile03) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      %c_rtpidx_2 = arith.constant 2 : index
      %c_rtpidx_3 = arith.constant 3 : index
      %c_rtpidx_4 = arith.constant 4 : index
      %c_rtpidx_5 = arith.constant 5 : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview3 = AIE.objectFifo.acquire @mtbuffer_0___MTout___rgba_rtp_thresh_1___in_buffer(Consume, 1) : !AIE.objectFifoSubview<memref<320xi32>>
         %elem3 = AIE.objectFifo.subview.access %subview3[0] : !AIE.objectFifoSubview<memref<320xi32>> -> memref<320xi32>
         %subview7 = AIE.objectFifo.acquire @rgba_rtp_thresh_1___out_buffer___mtbuffer_1___MTin(Produce, 1) : !AIE.objectFifoSubview<memref<320xi32>>
         %elem7 = AIE.objectFifo.subview.access %subview7[0] : !AIE.objectFifoSubview<memref<320xi32>> -> memref<320xi32>

         %nbytes = memref.load %rtp_0_3[%c_rtpidx_2] : memref<6xi32>
         %r_thresh = memref.load %rtp_0_3[%c_rtpidx_3] : memref<6xi32>
         %g_thresh = memref.load %rtp_0_3[%c_rtpidx_4] : memref<6xi32>
         %b_thresh = memref.load %rtp_0_3[%c_rtpidx_5] : memref<6xi32>
         func.call @rgba_rtp_thresh(%elem3, %elem7, %nbytes, %r_thresh, %g_thresh, %b_thresh) : (memref<320xi32>, memref<320xi32>, i32, i32, i32, i32) -> ()

         AIE.objectFifo.release @mtbuffer_0___MTout___rgba_rtp_thresh_1___in_buffer(Consume, 1)
         AIE.objectFifo.release @rgba_rtp_thresh_1___out_buffer___mtbuffer_1___MTin(Produce, 1)
      }
      AIE.end
   } { link_with="rgba_rtp_thresh.o" }

   AIE.core(%tile04) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      %c_rtpidx_2 = arith.constant 2 : index
      %c_rtpidx_3 = arith.constant 3 : index
      %c_rtpidx_4 = arith.constant 4 : index
      %c_rtpidx_5 = arith.constant 5 : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview4 = AIE.objectFifo.acquire @mtbuffer_0___MTout___rgba_rtp_thresh_2___in_buffer(Consume, 1) : !AIE.objectFifoSubview<memref<320xi32>>
         %elem4 = AIE.objectFifo.subview.access %subview4[0] : !AIE.objectFifoSubview<memref<320xi32>> -> memref<320xi32>
         %subview8 = AIE.objectFifo.acquire @rgba_rtp_thresh_2___out_buffer___mtbuffer_1___MTin(Produce, 1) : !AIE.objectFifoSubview<memref<320xi32>>
         %elem8 = AIE.objectFifo.subview.access %subview8[0] : !AIE.objectFifoSubview<memref<320xi32>> -> memref<320xi32>

         %nbytes = memref.load %rtp_0_4[%c_rtpidx_2] : memref<6xi32>
         %r_thresh = memref.load %rtp_0_4[%c_rtpidx_3] : memref<6xi32>
         %g_thresh = memref.load %rtp_0_4[%c_rtpidx_4] : memref<6xi32>
         %b_thresh = memref.load %rtp_0_4[%c_rtpidx_5] : memref<6xi32>
         func.call @rgba_rtp_thresh(%elem4, %elem8, %nbytes, %r_thresh, %g_thresh, %b_thresh) : (memref<320xi32>, memref<320xi32>, i32, i32, i32, i32) -> ()

         AIE.objectFifo.release @mtbuffer_0___MTout___rgba_rtp_thresh_2___in_buffer(Consume, 1)
         AIE.objectFifo.release @rgba_rtp_thresh_2___out_buffer___mtbuffer_1___MTin(Produce, 1)
      }
      AIE.end
   } { link_with="rgba_rtp_thresh.o" }

   AIE.core(%tile05) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      %c_rtpidx_2 = arith.constant 2 : index
      %c_rtpidx_3 = arith.constant 3 : index
      %c_rtpidx_4 = arith.constant 4 : index
      %c_rtpidx_5 = arith.constant 5 : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview5 = AIE.objectFifo.acquire @mtbuffer_0___MTout___rgba_rtp_thresh_3___in_buffer(Consume, 1) : !AIE.objectFifoSubview<memref<320xi32>>
         %elem5 = AIE.objectFifo.subview.access %subview5[0] : !AIE.objectFifoSubview<memref<320xi32>> -> memref<320xi32>
         %subview9 = AIE.objectFifo.acquire @rgba_rtp_thresh_3___out_buffer___mtbuffer_1___MTin(Produce, 1) : !AIE.objectFifoSubview<memref<320xi32>>
         %elem9 = AIE.objectFifo.subview.access %subview9[0] : !AIE.objectFifoSubview<memref<320xi32>> -> memref<320xi32>

         %nbytes = memref.load %rtp_0_5[%c_rtpidx_2] : memref<6xi32>
         %r_thresh = memref.load %rtp_0_5[%c_rtpidx_3] : memref<6xi32>
         %g_thresh = memref.load %rtp_0_5[%c_rtpidx_4] : memref<6xi32>
         %b_thresh = memref.load %rtp_0_5[%c_rtpidx_5] : memref<6xi32>
         func.call @rgba_rtp_thresh(%elem5, %elem9, %nbytes, %r_thresh, %g_thresh, %b_thresh) : (memref<320xi32>, memref<320xi32>, i32, i32, i32, i32) -> ()

         AIE.objectFifo.release @mtbuffer_0___MTout___rgba_rtp_thresh_3___in_buffer(Consume, 1)
         AIE.objectFifo.release @rgba_rtp_thresh_3___out_buffer___mtbuffer_1___MTin(Produce, 1)
      }
      AIE.end
   } { link_with="rgba_rtp_thresh.o" }

func.func @sequence(%itbuffer_0 : memref<720x1280xi32>,%itbuffer_1 : memref<720x1280xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c5120 = arith.constant 5120 : i32
    %c720 = arith.constant 720 : i32
    %c180 = arith.constant 180 : i32
    %c320 = arith.constant 320 : i32
    %c1280 = arith.constant 1280 : i32

    AIEX.ipu.rtp_write(0, 2, 2, 2) { buffer_sym_name = "rtp_0_2" }
    AIEX.ipu.rtp_write(0, 2, 3, 3) { buffer_sym_name = "rtp_0_2" }
    AIEX.ipu.rtp_write(0, 2, 4, 4) { buffer_sym_name = "rtp_0_2" }
    AIEX.ipu.rtp_write(0, 2, 5, 5) { buffer_sym_name = "rtp_0_2" }
    AIEX.ipu.rtp_write(0, 3, 2, 2) { buffer_sym_name = "rtp_0_3" }
    AIEX.ipu.rtp_write(0, 3, 3, 3) { buffer_sym_name = "rtp_0_3" }
    AIEX.ipu.rtp_write(0, 3, 4, 4) { buffer_sym_name = "rtp_0_3" }
    AIEX.ipu.rtp_write(0, 3, 5, 5) { buffer_sym_name = "rtp_0_3" }
    AIEX.ipu.rtp_write(0, 4, 2, 2) { buffer_sym_name = "rtp_0_4" }
    AIEX.ipu.rtp_write(0, 4, 3, 3) { buffer_sym_name = "rtp_0_4" }
    AIEX.ipu.rtp_write(0, 4, 4, 4) { buffer_sym_name = "rtp_0_4" }
    AIEX.ipu.rtp_write(0, 4, 5, 5) { buffer_sym_name = "rtp_0_4" }
    AIEX.ipu.rtp_write(0, 5, 2, 2) { buffer_sym_name = "rtp_0_5" }
    AIEX.ipu.rtp_write(0, 5, 3, 3) { buffer_sym_name = "rtp_0_5" }
    AIEX.ipu.rtp_write(0, 5, 4, 4) { buffer_sym_name = "rtp_0_5" }
    AIEX.ipu.rtp_write(0, 5, 5, 5) { buffer_sym_name = "rtp_0_5" }

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_1[%c0, %c0, %c0, %c0][%c1, %c1, %c720, %c1280][%c0, %c0, %c0]){ metadata= @mtbuffer_1___MTout___itbuffer_1___ITin, id = 1 : i32 } :(i32, i32, memref<720x1280xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_0[%c0, %c0, %c0, %c0][%c1, %c1, %c720, %c1280][%c0, %c0, %c0]){ metadata= @itbuffer_0___ITout___mtbuffer_0___MTin, id = 0 : i32 } :(i32, i32, memref<720x1280xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.sync {column = 0 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
    return
}
 }
}
