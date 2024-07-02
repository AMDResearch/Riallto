module  {
   AIE.device(ipu){

   %tile00 = AIE.tile(0, 0)
   %tile02 = AIE.tile(0, 2)
   %rtp_0_2 = AIE.buffer(%tile02) { sym_name = "rtp_0_2" } : memref<4xi32>
   AIE.objectFifo @itbuffer_0___ITout___threshold_0___in_buffer(%tile00, {%tile02}, 2 : i32) : !AIE.objectFifo<memref<1280xi32>>
   AIE.objectFifo @threshold_0___out_buffer___itbuffer_1___ITin(%tile02, {%tile00}, 2 : i32) : !AIE.objectFifo<memref<1280xi32>>


   func.func private @threshold(memref<1280xi32>, memref<1280xi32>, i32, i32) -> ()

   AIE.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      %c_rtpidx_2 = arith.constant 2 : index
      %c_rtpidx_3 = arith.constant 3 : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview1 = AIE.objectFifo.acquire @itbuffer_0___ITout___threshold_0___in_buffer(Consume, 1) : !AIE.objectFifoSubview<memref<1280xi32>>
         %elem1 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<1280xi32>> -> memref<1280xi32>
         %subview2 = AIE.objectFifo.acquire @threshold_0___out_buffer___itbuffer_1___ITin(Produce, 1) : !AIE.objectFifoSubview<memref<1280xi32>>
         %elem2 = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<1280xi32>> -> memref<1280xi32>

         %nbytes = memref.load %rtp_0_2[%c_rtpidx_2] : memref<4xi32>
         %r_threshold = memref.load %rtp_0_2[%c_rtpidx_3] : memref<4xi32>
         func.call @threshold(%elem1, %elem2, %nbytes, %r_threshold) : (memref<1280xi32>, memref<1280xi32>, i32, i32) -> ()

         AIE.objectFifo.release @itbuffer_0___ITout___threshold_0___in_buffer(Consume, 1)
         AIE.objectFifo.release @threshold_0___out_buffer___itbuffer_1___ITin(Produce, 1)
      }
      AIE.end
   } { link_with="threshold.o" }

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

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_1[%c0, %c0, %c0, %c0][%c1, %c1, %c720, %c1280][%c0, %c0, %c0]){ metadata= @threshold_0___out_buffer___itbuffer_1___ITin, id = 1 : i32 } :(i32, i32, memref<720x1280xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_0[%c0, %c0, %c0, %c0][%c1, %c1, %c720, %c1280][%c0, %c0, %c0]){ metadata= @itbuffer_0___ITout___threshold_0___in_buffer, id = 0 : i32 } :(i32, i32, memref<720x1280xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.sync {column = 0 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
    return
}
 }
}
