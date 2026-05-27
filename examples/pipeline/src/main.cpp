#include <type_traits>
#include <functional>
#include <exception>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <random>

#include <aff3ct.hpp>
#include "Decoder_LDPC_BP_flooding_gpu.hpp"
#include "Module/Channel/Channel_AWGN_LLR_gpu.hpp"
using namespace aff3ct;

struct params
{
    size_t n_threads = std::thread::hardware_concurrency();
    float  ebn0      = 2.50f; // SNR value
    float  R;                  // code rate (R=K/N)

    std::unique_ptr<factory::Source          > source;
    std::unique_ptr<factory::Codec_LDPC      > codec;
    std::unique_ptr<factory::Modem           > modem;
    std::unique_ptr<factory::Channel         > channel;
    std::unique_ptr<factory::Monitor_BFER    > monitor;
    std::unique_ptr<factory::Sink            > sink;
    std::unique_ptr<factory::Terminal        > terminal;
	std::unique_ptr<factory::Puncturer       > puncturer;
};
void init_params(int argc, char** argv, params &p);

struct modules
{
    std::unique_ptr<spu::module::Source<>>       source;
    std::unique_ptr<     module::Modem<>>        modem;
    std::unique_ptr<     module::Channel<>>      channel;
    std::unique_ptr<     module::Monitor_BFER<>> monitor;
    std::unique_ptr<spu::module::Sink<>>         sink;
    std::unique_ptr<     tools ::Codec_SIHO<>>   codec;
                         module::Encoder<>*      encoder;
                        module::Decoder_LDPC_BP_flooding_gpu<float, int>* decoder;
	std::unique_ptr<     module::Puncturer_5G<>> puncturer;
};
void init_modules(const params &p, modules &m);

struct utils
{
                std::unique_ptr<     tools  ::Sigma<>  > noise;     // a sigma noise type
    std::vector<std::unique_ptr<spu::tools  ::Reporter>> reporters; // list of reporters displayed in the terminal
                std::unique_ptr<spu::tools  ::Terminal > terminal;  // manage the output text in the terminal
                std::unique_ptr<spu::runtime::Pipeline>  pipeline;
};
void init_utils(const params &p, const modules &m, utils &u);

int main(int argc, char** argv)
{
    // StreamPU will catch and manage sigint
    spu::tools::Signal_handler::init();
	spu::Devices_manager::explore_gpu_devices();

    // get the AFF3CT version
    const std::string v = "v" + std::to_string(tools::version_major()) + "." +
                                std::to_string(tools::version_minor()) + "." +
                                std::to_string(tools::version_release());

    std::cout << "#----------------------------------------------------------"      << std::endl;
    std::cout << "# This is a basic program using the AFF3CT library (" << v << ")" << std::endl;
    std::cout << "# Feel free to improve it as you want to fit your needs."         << std::endl;
    std::cout << "#----------------------------------------------------------"      << std::endl;
    std::cout << "#"                                                                << std::endl;

    params  p; init_params (argc, argv, p); // create and initialize the parameters from the command line with factories
    modules m; init_modules(p, m         ); // create and initialize the modules

    // sockets binding (connect the sockets of the tasks = fill the input sockets with the output sockets)
    (*m.encoder)[      "encode::U_K"     ] = (*m.source )[   "generate::out_data" ];
	(*m.puncturer)[    "puncture::X_N1" ]  = (*m.encoder)[     "encode::X_N"      ];
    (*m.modem  )[    "modulate::X_N1"    ] = (*m.puncturer)[      "puncture::X_N2" ];
    (*m.channel)[   "add_noise::X_N"     ] = (*m.modem  )[   "modulate::X_N2"     ];
    (*m.modem  )[  "demodulate::Y_N1"    ] = (*m.channel)[  "add_noise::Y_N"      ];
	(*m.puncturer)[ "depuncture::Y_N1"   ] = (*m.modem  )[ "demodulate::Y_N2"     ];
    (*m.decoder)[ "decode_siho_gpu::Y_N" ]  = (*m.puncturer)[ "depuncture::Y_N2"    ];
    (*m.monitor)["check_errors::U"       ] = (*m.source )[   "generate::out_data" ];
    (*m.monitor)["check_errors::V"       ] = (*m.decoder)["decode_siho_gpu::V_K"      ];
    (*m.sink   )[  "send_count::in_count"] = (*m.source )[   "generate::out_count"];
    (*m.sink   )[  "send_count::in_data" ] = (*m.decoder)["decode_siho_gpu::V_K"      ];
    std::vector<float> sigma(1);
    (*m.channel)[   "add_noise::CP"      ] = sigma;
    (*m.modem  )[  "demodulate::CP"      ] = sigma;

	(*m.decoder)("decode_siho_gpu").set_execution_device_info({spu::device_interface::compute_api::CUDA, 0, 0}, true);

    utils u; init_utils(p, m, u); // create and initialize the utils

    // set the noise
    m.codec->set_noise(*u.noise);
    for (auto &m : u.pipeline->get_modules<tools::Interface_get_set_noise>())
        m->set_noise(*u.noise);

    // registering to noise updates
    u.noise->record_callback_update([&m](){ m.codec->notify_noise_update(); });
    for (auto &m : u.pipeline->get_modules<tools::Interface_notify_noise_update>())
        u.noise->record_callback_update([m](){ m->notify_noise_update(); });

    // set different seeds in the modules that uses PRNG
    std::mt19937 prng;
    for (auto &m : u.pipeline->get_modules<spu::tools::Interface_set_seed>())
        m->set_seed(prng());

    // compute the current sigma for the channel noise
    const auto esn0 = tools::ebn0_to_esn0(p.ebn0, p.R, p.modem->bps);
    std::fill(sigma.begin(), sigma.end(), tools::esn0_to_sigma(esn0, p.modem->cpm_upf));

    u.noise->set_values(sigma[0], p.ebn0, esn0);

    // display the performance (BER and FER) in real time (in a separate thread)
    u.terminal->legend();
    u.terminal->start_temp_report();

    // will automatically stop when `m.source->is_done()` will be `true` (end of input file) or if user press `Ctrl+c`
    u.pipeline->exec([&m]() -> bool
        {
            return m.monitor->is_done();
        });

    // display the performance (BER and FER) in the terminal
    u.terminal->final_report();

    // display the statistics of the tasks (if enabled)
    auto stages = u.pipeline->get_stages();
    for (size_t s = 0; s < stages.size(); s++)
    {
        const int n_threads = stages[s]->get_n_threads();
        std::cout << "#" << std::endl << "# Pipeline stage " << s << " (" << n_threads << " thread(s)): " << std::endl;
        spu::tools::Stats::show(stages[s]->get_tasks_per_types(), true);
    }
    std::cout << "#" << std::endl << "# End of the simulation" << std::endl;

    return 0;
}

void init_params(int argc, char** argv, params &p)
{
    p.source   = std::unique_ptr<factory::Source          >(new factory::Source          ());
    p.codec    = std::unique_ptr<factory::Codec_LDPC      >(new factory::Codec_LDPC      ());
    p.modem    = std::unique_ptr<factory::Modem           >(new factory::Modem           ());
    p.channel  = std::unique_ptr<factory::Channel         >(new factory::Channel         ());
    p.monitor  = std::unique_ptr<factory::Monitor_BFER    >(new factory::Monitor_BFER    ());
    p.sink     = std::unique_ptr<factory::Sink            >(new factory::Sink            ());
    p.terminal = std::unique_ptr<factory::Terminal        >(new factory::Terminal        ());
	p.puncturer = std::unique_ptr<factory::Puncturer      >(new factory::Puncturer       ());

    std::vector<factory::Factory*> params_list = { p.source  .get(), p.codec  .get(), p.puncturer.get(), p.modem.get(),
                                                   p.channel .get(), p.monitor.get(), p.sink .get(),
                                                   p.terminal.get()                                  };

    // parse the command for the given parameters and fill them
    tools::Command_parser cp(argc, argv, params_list, true);
    if (cp.parsing_failed())
    {
        cp.print_help    ();
        cp.print_warnings();
        cp.print_errors  ();
        std::exit(1);
    }

    std::cout << "# Simulation parameters: " << std::endl;
    tools::Header::print_parameters(params_list); // display the headers (= print the AFF3CT parameters on the screen)
    std::cout << "#" << std::endl;
    cp.print_warnings();

     p.R = (float)p.codec->enc->K / (float)p.puncturer->N; // compute the code rate
}

void init_modules(const params &p, modules &m)
{
    m.source  = std::unique_ptr<spu::module::Source      <>>(p.source ->build());
    m.codec   = std::unique_ptr<     tools ::Codec_SIHO  <>>(p.codec  ->build());
    m.modem   = std::unique_ptr<     module::Modem       <>>(p.modem  ->build());
    m.channel = std::unique_ptr<     module::Channel     <>>(p.channel->build());
    m.monitor = std::unique_ptr<     module::Monitor_BFER<>>(p.monitor->build());
    m.sink    = std::unique_ptr<spu::module::Sink        <>>(p.sink   ->build());
    m.encoder = &m.codec->get_encoder();
    m.decoder = new aff3ct::module::Decoder_LDPC_BP_flooding_gpu<float, int>(p.codec.get()->K, p.codec->enc.get()->N_cw, p.puncturer.get()->N, 20);
	// Building the puncturer
	std::vector<bool> pct_pattern;
    m.puncturer = std::unique_ptr<module::Puncturer_5G<>> (new module::Puncturer_5G <int, float> (p.puncturer.get()->K, p.puncturer.get()->N,  p.codec->enc.get()->N_cw,  pct_pattern));

    // prevent the monitor to stop the pipeline
    //m.monitor->disable_is_done(true);
}

void init_utils(const params &p, const modules &m, utils &u)
{
    u.pipeline.reset(spu::tools::Pipeline_builder()
        .add_task_for_checking((*m.source)("generate")) //     first task to build the sequence and check with pipeline
        // ------------------------------------------------------------------------------------------------------------
        .add_stage(spu::tools::Pipeline_builder::Stage_builder() // ------------------------------------------- STAGE 0
            .add_first_task((*m.source)("generate")) //                                         first task of the stage
            .add_last_task((*m.source)("generate")) //                                          last  task of the stage
            .set_n_threads(1)) //                                               run on a single thread (NO replication)
        // ------------------------------------------------------------------------------------------------------------
        .configure_interstage_synchro(spu::tools::Pipeline_builder::Synchro_builder() // ---------- INTER-STAGE 0 <-> 1
            .set_buffer_size(3) //                                                          synchronization buffer size
            .set_active_waiting(false)) //                                          passive waiting for synchronization
        // ------------------------------------------------------------------------------------------------------------
        .add_stage(spu::tools::Pipeline_builder::Stage_builder() // ------------------------------------------- STAGE 1
            .add_first_task((*m.encoder)("encode")) //                                          first task of the stage
            .add_last_task((*m.puncturer)("depuncture")) //                                      last  task of the stage
            .set_n_threads(4)) //          can run on a multiple threads (with replication)
        // ------------------------------------------------------------------------------------------------------------
        .configure_interstage_synchro(spu::tools::Pipeline_builder::Synchro_builder() // ---------- INTER-STAGE 1 <-> 2
            .set_buffer_size(3) //                                                          synchronization buffer size
            .set_active_waiting(false)) //                                          passive waiting for synchronization

		.add_stage(spu::tools::Pipeline_builder::Stage_builder() // ------------------------------------------- STAGE 1
            .add_first_task((*m.decoder)("decode_siho_gpu")) //                                          first task of the stage
            .add_last_task((*m.decoder)("decode_siho_gpu")) //                                      last  task of the stage
            .set_n_threads(2)) //          can run on a multiple threads (with replication)
        // ------------------------------------------------------------------------------------------------------------
        .configure_interstage_synchro(spu::tools::Pipeline_builder::Synchro_builder() // ---------- INTER-STAGE 1 <-> 2
            .set_buffer_size(3) //                                                          synchronization buffer size
            .set_active_waiting(false)) //                                          passive waiting for synchronization
        // ------------------------------------------------------------------------------------------------------------
        .add_stage(spu::tools::Pipeline_builder::Stage_builder() // ------------------------------------------- STAGE 2
            .set_first_tasks({&(*m.monitor)("check_errors"), //                            two first tasks of the stage
                              &(*m.sink)("send_count")}) //                  (and NO need for a last task in this case)
            .set_n_threads(1)) //                                               run on a single thread (NO replication)
        // ------------------------------------------------------------------------------------------------------------
        .build_ptr()); //               finally allocate and initialize a new pipeline from all the previous parameters

    std::ofstream f("pipeline.dot");
    u.pipeline->export_dot(f);

    // create a sigma noise type
    u.noise = std::unique_ptr<tools::Sigma<>>(new tools::Sigma<>());
    // report the noise values (Es/N0 and Eb/N0)
    u.reporters.push_back(std::unique_ptr<spu::tools::Reporter>(new tools::Reporter_noise<>(*u.noise)));
    // report the bit/frame error rates
    u.reporters.push_back(std::unique_ptr<spu::tools::Reporter>(new tools::Reporter_BFER<>(*m.monitor)));
    // report the simulation throughputs
    u.reporters.push_back(std::unique_ptr<spu::tools::Reporter>(new tools::Reporter_throughput<>(*m.monitor)));
    // create a terminal that will display the collected data from the reporters
    u.terminal = std::unique_ptr<spu::tools::Terminal>(p.terminal->build(u.reporters));

    // configuration of the pipeline tasks
    for (auto& type : u.pipeline->get_tasks_per_types()) for (auto& tsk : type)
    {
        tsk->set_debug      (false); // disable the debug mode
        tsk->set_debug_limit(16   ); // display only the 16 first bits if the debug mode is enabled
        tsk->set_stats      (true ); // enable the statistics

        // enable the fast mode (= disable the useless verifs in the tasks) if there is no debug and stats modes
        if (!tsk->is_debug() && !tsk->is_stats())
            tsk->set_fast(true);
    }
}