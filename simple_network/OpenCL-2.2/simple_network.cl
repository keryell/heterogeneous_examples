/* Small OpenCL C++ 2.2 example to experiment on how a network
   application on FPGA by using only C++ and built-in kernels would
   look like

   - use clean C++ (only 1 pointer * used in 1 kernel...)

   - use only standard (non blocking) pipes

   - use static object constructors to start long-running kernel
     without host action

   - use only built-in kernels to access specific hardware features
     expected to come from a Xilinx library for example
*/

#include <opencl_device_queue>
#include <opencl_memory>
#include <opencl_pipe>

/* Some libraries providing functions, classes and built-in RTL
   kernels to control the platform */
#include <xilinx/interruption>
#include <xilinx/networking>
#include <xilinx/util>

/* This dummy kernel is just here to force programm loading and
   program-scope object initizalization when it is run.

   The interesting side effect is starting the kernel graphs on the
   accelerator
*/
kernel void force_init() {
}


// To read interrupt information from the interrupt controller
cl::pipe_storage<xlnx::interrupt::descriptor, 1> interrupt_channel;
/* Instantiate an interrupt controller which sends interrupt
   descriptions on the provided pipe

   Behind the scene, the constructor just launchs a built-in RTL
   kernel from the DSA in the back-ground with a writing pipe as
   parameter constructed from the provided pipe_storage.

   We could provide a higher-level interface, such as hiding the pipe
   to the used in some methods.
*/
xlnx::interrupt::controller interrupt_controller { interrupt_channel };

/* Can store 1 IEEE802 packet


   In a real application, probably we would not use such a big
   granularity but this is a simple example. */
cl::pipe_storage<xlnx::network::ethernet::packet, 1> eth0_packet_channel,
  eth1_packet_channel, raw_eth0_packet, raw_eth1_packet;


/* Some "wires" to start eth0 and eth1 processing from the interrupt
   dispatcher

   By itself there is nothing to really propagate, just the ready
   status. But since it is not possible (yet) to send a void like in
   std::future, just send a minimal value: a boolean.

   The depth can be chosen to set the number of interruptions in
   fly. Here only 1.
*/
cl::pipe_storage<std::bool, 1> trigger_eth0_reading, trigger_eth1_writing;


/* Instantiate the external Ethernet interface built-in kernels
   provided by the DSA and connect each one to its pipe

   Actually we could provide a higher-level interface hiding the pipe in
   some methods.
*/
xlnx::network::controller::eth0 eth0_controller { raw_eth0_packet };
xlnx::network::controller::eth1 eth1_controller { raw_eth1_packet };

/* A function to implement a read on some storage_pipe that waits up
   to success

   Of course it assumes a memory model and an IFP guarantee typical on
   FPGA... Otherwise it could use blocking pipe extension.

   This could be a trivial library from Khronos.
*/
auto blocking_read = [] (auto some_pipe_storage, auto &a_variable) {
  /* By default make_pipe returns a read-access pipe but let's be
     explicit for educational purpose */
  auto reader = cl::make_pipe<cl::pipe_access::read>(some_pipe_storage);
  // Spin up to successful read
  while (!reader.read(a_variable))
    ;
};


/* A function to implement a write on some storage_pipe that waits up
   to success

   Of course it assumes a memory model and an IFP guarantee typical on
   FPGA... Otherwise it could use blocking pipe extension.

   This could be a trivial library from Khronos.
*/
auto blocking_write = [] (auto some_pipe_storage, auto const &a_variable) {
  /* By default make_pipe returns a read-access pipe but let's be
     explicit for educational purpose */
  auto writer = cl::make_pipe<cl::pipe_access::write>(some_pipe_storage);
  // Spin up to successful write
  while (!writer.write(a_variable))
    ;
};


/* A long running kernel that reads the interrupts and dispatch the
   action

   This a contrived example to demonstrate how to use interruptions
*/
kernel void dispatch_interrupt() {
  xlnx::interrupt_descriptor i;
  for (; /* ever */ ;) {
    // Wait for the pipe controlled by the interrupt controller
    blocking_read(interrupt_channel, i);
    switch (i.source) {
    case xlnx::device::eth0:
      // Send a ready signal to eth0_receiver
      blocking_write(trigger_eth0_reading, true);
      break;
    case xlnx::device::eth1:
      // Send a ready signal to eth1_sender
      blocking_write(trigger_eth1_writing, true);
      break;
    }
  }
}


/* Start an asynchronous kernel without any argument on 1
   work-item from the device

   The typical use case is to be used as a program scope variable to
   start kernels.

   This could be a trivial library from Khronos.

   Assume cl::get_default_device_queue() is working out-of-the box or
   that the host has already created it.
*/
struct start_kernel {
  template <typename Kernel>
  start_kernel(Kernel k) {
    cl::get_default_device_queue().enqueue_kernel(cl::enqueue_policy::no_wait,
                                                  { 1 },
                                                  k);
  }
};


// Start dispatch_interrupt at program level to listen to the interrupts
start_kernel launch_interrupt_dispatcher { dispatch_interrupt };


/* Read an ethernet packet from eth0 and send it to the forwarder */
kernel void eth0_receiver() {
  for (; /* ever */ ;) {
    std::bool unused;
    blocking_read(trigger_eth0_reading, unused);
    /* Since we have been notified by interruption, we know the pipe
       from Ethernet is ready, so no need to wait */
    xlnx::ethernet::packet p;
    cl::make_pipe<cl::pipe_access::read>(raw_eth0_packet).read(p);
    // Add some buffering code here for better buffer bloat :-)
    // [...]
    // Send a packet to the router
    blocking_write(eth0_packet_channel, p);
  }
}


// Start the eth0 receiver at program level
start_kernel launch_eth0_receiver { eth0_receiver };


/* Read an ethernet packet from the forwarder and send it to eth1 */
kernel void eth1_sender() {
  for (; /* ever */ ;) {
    std::bool unused;
    blocking_read(trigger_eth1_writing, unused);
    xlnx::ethernet::packet p;
    // Wait for a packet to forward
    blocking_read(eth1_packet_channel, p);
    /* Since we have been notified by interruption, we know the pipe
       to Ethernet is ready, so no need to wait */
    cl::make_pipe<cl::pipe_access::write>(raw_eth1_packet).write(p);
  }
}


// Start the eth1 sender at program level
start_kernel launch_eth1_sender { eth1_sender };

/* Use a set implementation with static memory allocation to implement
   the forwarding table for each address. Store up to 10000
   addresses */
using forward_t = xlnx::util::set<xlnx::network::ethernet::address, 1000>;
forward_t forward;
// A lock to protect the access to the forwarding table
cl:: atomic_flag forward_lock = ATOMIC_FLAG_INIT;


// A trivial L2 packet forwarder from eth0 to eth1
kernel void L2_router() {
  xlnx::ethernet::packet p;
  for (; /* ever */ ;) {
    blocking_read(eth0_packet_channel, p);
    /* If the packet is to be forwarded to eth1 according to the
       destination IEEE802 address, just do it!

       But the global table may be updated by the host at the same
       time, so acquire a lock on the table first
    */
    while(forward_lock.test_and_set())
      ;
    //  Is the address in the forward set?
    std::bool forward_p = forward.count(p.dest);
    // Release the lock as early as possible
    forward_lock.clear();
    // Then do the real forwarding if required
    if (forward_p)
      blocking_write(eth1_packet_channel, p);
  }
}


/* Update the forwarding table

   Since there is no host-side access to program scope memories or
   pipes, use a proxy-kernel to copy the information from the host
   through a global buffer.

   Using a direct access from the host to program scope buffer would
   remove the copy and the latency, but how to deal with the lock
   without launching 2 kernels?
*/
kernel void update_forward_table(cl::global_ptr<forward_t> new_table) {
  // Lock the table
  while(forward_lock.test_and_set())
    ;
  // Massive update
  forward = *new_table;
  // Release the lock
  forward_lock.clear();
}
