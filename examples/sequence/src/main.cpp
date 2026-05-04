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
#include "Decoder_LDPC_BP_flooding_cuda.hpp"
using namespace aff3ct;

// #define STEP_BY_STEP

struct params
{
#ifndef STEP_BY_STEP
    size_t n_threads = 1;
#else
    size_t n_threads = 1;
#endif
    float  ebn0_min  =  0.00f; // minimum SNR value
    float  ebn0_max  =  4.01f; // maximum SNR value
    float  ebn0_step =  0.50f; // SNR step
    float  R;                  // code rate (R=K/N)

    std::unique_ptr<factory::Source          > source;
    std::unique_ptr<factory::Codec_LDPC      > codec;
    std::unique_ptr<factory::Modem           > modem;
    std::unique_ptr<factory::Channel         > channel;
    std::unique_ptr<factory::Monitor_BFER    > monitor;
    std::unique_ptr<factory::Terminal        > terminal;
	std::unique_ptr<factory::Puncturer       > puncturer;
	std::unique_ptr<factory::Quantizer       > quantizer;
};
void init_params(int argc, char** argv, params &p);

struct modules
{
    std::unique_ptr<spu::module::Source<>>       source;
    std::unique_ptr<     module::Modem<>>        modem;
    std::unique_ptr<     module::Channel<>>      channel;
    std::unique_ptr<     module::Monitor_BFER<>> monitor;
    std::unique_ptr<     tools ::Codec_LDPC<>>   codec;
                         module::Encoder<>*      encoder;
                         module::Decoder_LDPC_BP_flooding_cuda<int8_t, int>* decoder;
	std::unique_ptr<     module::Puncturer_5G<>> puncturer;
	std::unique_ptr<     module::Quantizer_pow2_fast<float, int8_t>> quantizer;
};
void init_modules(const params &p, modules &m);

namespace aff3ct { namespace tools {
using Monitor_BFER_reduction = Monitor_reduction<module::Monitor_BFER<>>;
} }

struct utils
{
                std::unique_ptr<       tools::Sigma<>               >  noise;       // a sigma noise type
    std::vector<std::unique_ptr<spu::  tools::Reporter              >> reporters;   // list of reporters displayed in the terminal
                std::unique_ptr<spu::  tools::Terminal              >  terminal;    // manage the output text in the terminal
                std::unique_ptr<       tools::Monitor_BFER_reduction>  monitor_red; // main monitor object that reduces all the thread monitors
                std::unique_ptr<spu::runtime::Sequence              >  sequence;    // a sequence to run the processing chain
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
    (*m.encoder)[      "encode::U_K" ]      = (*m.source )[   "generate::out_data"];
	(*m.puncturer)[      "puncture::X_N1" ] = (*m.encoder)[     "encode::X_N"     ];
    (*m.modem  )[    "modulate::X_N1"]      = (*m.puncturer)[      "puncture::X_N2" ];
    (*m.channel)[   "add_noise::X_N" ]      = (*m.modem  )[   "modulate::X_N2"    ];
    (*m.modem  )[  "demodulate::Y_N1"]      = (*m.channel)[  "add_noise::Y_N"     ];
	(*m.puncturer)[    "depuncture::Y_N1" ]  = (*m.modem  )[ "demodulate::Y_N2"    ];
	(*m.quantizer)[      "process::Y_N1" ]  = (*m.puncturer)[ "depuncture::Y_N2"    ];
    (*m.decoder)[ "decode_siho_cuda::Y_N" ] =  (*m.quantizer)[      "process::Y_N2" ];
    (*m.monitor)["check_errors::U"   ]      = (*m.source )[   "generate::out_data"];
    (*m.monitor)["check_errors::V"   ]      = (*m.decoder)["decode_siho_cuda::V_K" ];
    std::vector<float> sigma(1);
    (*m.channel)[   "add_noise::CP"  ] = sigma;
    (*m.modem  )[  "demodulate::CP"  ] = sigma;

    utils u; init_utils(p, m, u); // create and initialize the utils

    // set the noise
    m.codec->set_noise(*u.noise);
    for (auto &m : u.sequence->get_modules<tools::Interface_get_set_noise>())
        m->set_noise(*u.noise);

    // registering to noise updates
    u.noise->record_callback_update([&m](){ m.codec->notify_noise_update(); });
    for (auto &m : u.sequence->get_modules<tools::Interface_notify_noise_update>())
        u.noise->record_callback_update([m](){ m->notify_noise_update(); });

    // set different seeds in the modules that uses PRNG
    std::mt19937 prng;
    for (auto &m : u.sequence->get_modules<spu::tools::Interface_set_seed>())
        m->set_seed(prng());

    // display the legend in the terminal
    u.terminal->legend();

    // loop over the various SNRs
    for (auto ebn0 = p.ebn0_min; ebn0 < p.ebn0_max; ebn0 += p.ebn0_step)
    {
        // compute the current sigma for the channel noise
        const auto esn0 = tools::ebn0_to_esn0(ebn0, p.R, p.modem->bps);
        std::fill(sigma.begin(), sigma.end(), tools::esn0_to_sigma(esn0, p.modem->cpm_upf));

        u.noise->set_values(sigma[0], ebn0, esn0);

        // display the performance (BER and FER) in real time (in a separate thread)
        u.terminal->start_temp_report();

        // execute the simulation sequence (multi-threaded)
#ifndef STEP_BY_STEP
        u.sequence->exec([&u]() -> bool
        {
            return u.monitor_red->is_done();
        });
#else
        spu::runtime::Task* cur_task;
        do
            while ((cur_task = u.sequence->exec_step()));
            // {
            //     std::cout << "cur_task->get_name() = " << cur_task->get_name() << std::endl;
            // }
        while (!u.sequence->is_done() && !u.monitor_red->is_done() && !spu::tools::Signal_handler::is_sigint());
#endif

        // final reduction
        u.monitor_red->reduce();

        // display the performance (BER and FER) in the terminal
        u.terminal->final_report();

        // reset the monitors for the next SNR
        u.monitor_red->reset();
    }

    // display the statistics of the tasks (if enabled)
    std::cout << "#" << std::endl;
    spu::tools::Stats::show(u.sequence->get_modules_per_types(), true);
    std::cout << "# End of the simulation" << std::endl;

    return 0;
}

void init_params(int argc, char** argv, params &p)
{
    p.source   = std::unique_ptr<factory::Source          >(new factory::Source          ());
    p.codec    = std::unique_ptr<factory::Codec_LDPC      >(new factory::Codec_LDPC      ());
    p.modem    = std::unique_ptr<factory::Modem           >(new factory::Modem           ());
    p.channel  = std::unique_ptr<factory::Channel         >(new factory::Channel         ());
    p.monitor  = std::unique_ptr<factory::Monitor_BFER    >(new factory::Monitor_BFER    ());
    p.terminal = std::unique_ptr<factory::Terminal        >(new factory::Terminal        ());
	p.puncturer = std::unique_ptr<factory::Puncturer      >(new factory::Puncturer       ());
	p.quantizer = std::unique_ptr<factory::Quantizer      >(new factory::Quantizer       ());

    std::vector<factory::Factory*> params_list = { p.source .get(), p.codec  .get(), p.puncturer .get(), p.modem   .get(),
                                                   p.channel.get(), p.monitor.get(), p.terminal.get(), p.quantizer.get() };

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
    m.source  = std::unique_ptr<spu::module::Source      <>> (p.source ->build());
    m.codec   = std::unique_ptr<     tools ::Codec_LDPC  <>> (p.codec  ->build());
	m.modem   = std::unique_ptr<     module::Modem       <>>(p.modem  ->build());
    m.channel = std::unique_ptr<     module::Channel     <>> (p.channel->build());
    m.monitor = std::unique_ptr<     module::Monitor_BFER<>> (p.monitor->build());
    m.encoder = &m.codec->get_encoder();
    m.decoder = new aff3ct::module::Decoder_LDPC_BP_flooding_cuda<int8_t, int>(p.codec.get()->K, p.codec->enc.get()->N_cw, p.puncturer.get()->N, 20);
	
	std::vector<bool> pct_pattern;
    m.puncturer = std::unique_ptr<module::Puncturer_5G<>> (new module::Puncturer_5G <int, float> (p.puncturer.get()->K, p.puncturer.get()->N,  p.codec->enc.get()->N_cw,  pct_pattern));
    m.quantizer = std::unique_ptr<module::Quantizer_pow2_fast<float, int8_t>> (new module::Quantizer_pow2_fast<float, int8_t>(p.codec->enc.get()->N_cw, 3));
}

void init_utils(const params &p, const modules &m, utils &u)
{
    // create a sequence, automatically replicated on 4 `p.n_threads` threads
    u.sequence = std::unique_ptr<spu::runtime::Sequence>(new spu::runtime::Sequence((*m.source)("generate"),
        p.n_threads ? p.n_threads : 1));
    // allocate a common monitor module to reduce all the monitors
    u.monitor_red = std::unique_ptr<tools::Monitor_BFER_reduction>(new tools::Monitor_BFER_reduction(
        u.sequence->get_modules<module::Monitor_BFER<>>()));
    u.monitor_red->set_reduce_frequency(std::chrono::milliseconds(500));
    // create a sigma noise type
    u.noise = std::unique_ptr<tools::Sigma<>>(new tools::Sigma<>());
    // report the noise values (Es/N0 and Eb/N0)
    u.reporters.push_back(std::unique_ptr<spu::tools::Reporter>(new tools::Reporter_noise<>(*u.noise)));
    // report the bit/frame error rates
    u.reporters.push_back(std::unique_ptr<spu::tools::Reporter>(new tools::Reporter_BFER<>(*u.monitor_red)));
    // report the simulation throughputs
    u.reporters.push_back(std::unique_ptr<spu::tools::Reporter>(new tools::Reporter_throughput<>(*u.monitor_red)));
    // create a terminal that will display the collected data from the reporters
    u.terminal = std::unique_ptr<spu::tools::Terminal>(p.terminal->build(u.reporters));

    // configuration of the sequence tasks
    for (auto& mod : u.sequence->get_modules<spu::module::Module>(false))
        for (auto& tsk : mod->tasks)
        {
            tsk->set_debug      (false); // disable the debug mode
            tsk->set_debug_limit(16   ); // display only the 16 first bits if the debug mode is enabled
            tsk->set_stats      (true ); // enable the statistics

            // enable the fast mode (= disable the useless verifs in the tasks) if there is no debug and stats modes
            if (!tsk->is_debug() && !tsk->is_stats())
                tsk->set_fast(true);
        }
}