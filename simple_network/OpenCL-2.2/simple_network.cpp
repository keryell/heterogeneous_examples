/** Host part of the simple networking application

    This starts the L2 forwarding application and loop on updating the
    forwarding table according to some external user-interface.
 */

#include <boost/compute.hpp>
#include <xilinx/networking>
#include <xilinx/util>

/* This is an imaginary user interface to command the system to be
   implemented
*/
class user_interface {

public:

  template <typename T>
  void update(T &forward) {
    // [...] Update the forwarding table somehow...
  }

};

user_interface ux;


int main() {
  // Create the OpenCL context to attach resources on the device
  auto context = boost::compute::system::default_context();
  // Create the OpenCL command queue to control the device
  auto command_queue = boost::compute::system::default_queue();

  /* Create a device default queue so a kernel can enqueue anther kernel.

     Note that program-scope and static constructors are actually run
     by an OpenCL run-time initializing kernel with 1 work-item, so
     they can launch kernels.
   */
  boost::compute::command_queue dq { context,
      command_queue.get_device(),
      boost::compute::command_queue::enable_out_of_order_execution
      | boost::compute::command_queue::on_device
      | boost::compute::command_queue::on_device_default };

  /* Construct an OpenCL program from the source string

     Boost.Compute caches the binary, so on FPGA you pay the lengthy
     compilation only... once. */
  auto program = boost::compute::program::create_with_source_file
    ("simple_network.cl", boost::compute::system::default_context());

  program.build();

  /* Start the dummy kernel force_init just to initialize the whole
     program on the accelerator, which start the whole application by
     side effect
  */
  auto force_init = boost::compute::kernel { program, "force_init" };

  /* Launch force_init kernel with 1 work-item, forcing the
     initialization of the program-scope objects */
  command_queue.enqueue_task(force_init);

  auto update = boost::compute::kernel { program, "update_forward_table" };

  // The forwarding table
  xlnx::util::set<xlnx::network::ethernet::address, 1000> forward;
  // And the buffer to send it to the device
  boost::compute::buffer fb { context, sizeof(forward), CL_MEM_READ_ONLY };
  // The update_forward_table kernel will take this buffer
  update.set_args(fb);
 
  for (; /* ever */ ;) {
    // Get some forwarding update from some external user interface...
    ux.update(forward);
    // Send the forwarding table top the accelerator
    command_queue.enqueue_write_buffer(fb, 0 /* Offset */,
				       sizeof(forward), &forward);
    /* Launch the update_forward_table kernel with 1 work-item to lock
       and update the table on the device */
    command_queue.enqueue_task(update);
  }
}
