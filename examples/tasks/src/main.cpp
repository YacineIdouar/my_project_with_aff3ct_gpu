#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include <aff3ct.hpp>
using namespace aff3ct;

struct params
{
    int   K         =  32;     // number of information bits
    int   N         = 128;     // codeword size
    int   fe        = 100;     // number of frame errors
    int   seed      =   0;     // PRNG seed for the AWGN channel
    float ebn0_min  =   0.00f; // minimum SNR value
    float ebn0_max  =  10.01f; // maximum SNR value
    float ebn0_step =   1.00f; // SNR step
    float R;                   // code rate (R=K/N)
};
void init_params(params &p);

struct modules
{
    std::unique_ptr<spu::module::Source_random<>>          source;
    std::unique_ptr<     module::Encoder_repetition_sys<>> encoder;
    std::unique_ptr<     module::Modem_BPSK<>>             modem;
    std::unique_ptr<     module::Channel_AWGN_LLR<>>       channel;
    std::unique_ptr<     module::Decoder_repetition_std<>> decoder;
    std::unique_ptr<     module::Monitor_BFER<>>           monitor;
    std::vector<const spu::module::Module*>                list; // list of module pointers declared in this structure
};
void init_modules(const params &p, modules &m);

struct utils
{
    std::unique_ptr<tools::Sigma<>>                    noise;     // a sigma noise type
    std::vector<std::unique_ptr<spu::tools::Reporter>> reporters; // list of reporters dispayed in the terminal
    std::unique_ptr<spu::tools::Terminal_std>          terminal;  // manage the output text in the terminal
};
void init_utils(const modules &m, utils &u);

int main(int argc, char** argv)
{
    // enable auto allocation of the output socket which is not the default now, because the allocation are made in
    // the sequence (BUT there is no sequence here)
    spu::tools::Buffer_allocator::set_task_autoalloc(true);

    // get the AFF3CT version
    const std::string v = "v" + std::to_string(tools::version_major()) + "." +
                                std::to_string(tools::version_minor()) + "." +
                                std::to_string(tools::version_release());

    std::cout << "#----------------------------------------------------------"      << std::endl;
    std::cout << "# This is a basic program using the AFF3CT library (" << v << ")" << std::endl;
    std::cout << "# Feel free to improve it as you want to fit your needs."         << std::endl;
    std::cout << "#----------------------------------------------------------"      << std::endl;
    std::cout << "#"                                                                << std::endl;

    params  p; init_params (p   ); // create and initialize the parameters defined by the user
    modules m; init_modules(p, m); // create and initialize the modules
    utils   u; init_utils  (m, u); // create and initialize the utils

    // display the legend in the terminal
    u.terminal->legend();

    // sockets binding (connect the sockets of the tasks = fill the input sockets with the output sockets)
    using namespace module;
    (*m.encoder)[      "encode::U_K" ] = (*m.source )[   "generate::out_data"];
    (*m.modem  )[    "modulate::X_N1"] = (*m.encoder)[     "encode::X_N"     ];
    (*m.channel)[   "add_noise::X_N" ] = (*m.modem  )[   "modulate::X_N2"    ];
    (*m.modem  )[  "demodulate::Y_N1"] = (*m.channel)[  "add_noise::Y_N"     ];
    (*m.decoder)[ "decode_siho::Y_N" ] = (*m.modem  )[ "demodulate::Y_N2"    ];
    (*m.monitor)["check_errors::U"   ] = (*m.source )[   "generate::out_data"];
    (*m.monitor)["check_errors::V"   ] = (*m.decoder)["decode_siho::V_K"     ];

    std::vector<float> sigma(1);
    (*m.channel)[ "add_noise::CP"] = sigma;
    (*m.modem  )["demodulate::CP"] = sigma;

    // loop over the various SNRs
    for (auto ebn0 = p.ebn0_min; ebn0 < p.ebn0_max; ebn0 += p.ebn0_step)
    {
        // compute the current sigma for the channel noise
        const auto esn0 = tools::ebn0_to_esn0(ebn0, p.R);
        std::fill(sigma.begin(), sigma.end(), tools::esn0_to_sigma(esn0));

        u.noise->set_values(sigma[0], ebn0, esn0);

        // display the performance (BER and FER) in real time (in a separate thread)
        u.terminal->start_temp_report();

        // run the simulation chain
        while (!m.monitor->fe_limit_achieved() && !spu::tools::Signal_handler::is_sigint())
        {
            // it is important to use enumerate (instead strings) to get the tasks for performance reasons
            (*m.source )[spu::module::src::tsk::generate    ].exec();
            (*m.encoder)[             enc::tsk::encode      ].exec();
            (*m.modem  )[             mdm::tsk::modulate    ].exec();
            (*m.channel)[             chn::tsk::add_noise   ].exec();
            (*m.modem  )[             mdm::tsk::demodulate  ].exec();
            (*m.decoder)[             dec::tsk::decode_siho ].exec();
            (*m.monitor)[             mnt::tsk::check_errors].exec();
        }

        // reset sigint if previously triggered
        if (spu::tools::Signal_handler::is_sigint())
            spu::tools::Signal_handler::reset_sigint();

        // display the performance (BER and FER) in the terminal
        u.terminal->final_report();

        // reset the monitor and the terminal for the next SNR
        m.monitor->reset();
    }

    // display the statistics of the tasks (if enabled)
    std::cout << "#" << std::endl;
    spu::tools::Stats::show(m.list, true);
    std::cout << "# End of the simulation" << std::endl;

    return 0;
}

void init_params(params &p)
{
    p.R = (float)p.K / (float)p.N;
    std::cout << "# * Simulation parameters: "              << std::endl;
    std::cout << "#    ** Frame errors   = " << p.fe        << std::endl;
    std::cout << "#    ** Noise seed     = " << p.seed      << std::endl;
    std::cout << "#    ** Info. bits (K) = " << p.K         << std::endl;
    std::cout << "#    ** Frame size (N) = " << p.N         << std::endl;
    std::cout << "#    ** Code rate  (R) = " << p.R         << std::endl;
    std::cout << "#    ** SNR min   (dB) = " << p.ebn0_min  << std::endl;
    std::cout << "#    ** SNR max   (dB) = " << p.ebn0_max  << std::endl;
    std::cout << "#    ** SNR step  (dB) = " << p.ebn0_step << std::endl;
    std::cout << "#"                                        << std::endl;
}

void init_modules(const params &p, modules &m)
{
    m.source  = std::unique_ptr<spu::module::Source_random         <>>(new spu::module::Source_random         <>(p.K      ));
    m.encoder = std::unique_ptr<     module::Encoder_repetition_sys<>>(new      module::Encoder_repetition_sys<>(p.K, p.N ));
    m.modem   = std::unique_ptr<     module::Modem_BPSK            <>>(new      module::Modem_BPSK            <>(p.N      ));
    m.channel = std::unique_ptr<     module::Channel_AWGN_LLR      <>>(new      module::Channel_AWGN_LLR      <>(p.N      ));
    m.decoder = std::unique_ptr<     module::Decoder_repetition_std<>>(new      module::Decoder_repetition_std<>(p.K, p.N ));
    m.monitor = std::unique_ptr<     module::Monitor_BFER          <>>(new      module::Monitor_BFER          <>(p.K, p.fe));
    m.channel->set_seed(p.seed);

    m.list = { m.source.get(), m.encoder.get(), m.modem.get(), m.channel.get(), m.decoder.get(), m.monitor.get() };

    // configuration of the module tasks
    for (auto& mod : m.list)
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

void init_utils(const modules &m, utils &u)
{
    // create a sigma noise type
    u.noise = std::unique_ptr<tools::Sigma<>>(new tools::Sigma<>());
    // report the noise values (Es/N0 and Eb/N0)
    u.reporters.push_back(std::unique_ptr<spu::tools::Reporter>(new tools::Reporter_noise<>(*u.noise)));
    // report the bit/frame error rates
    u.reporters.push_back(std::unique_ptr<spu::tools::Reporter>(new tools::Reporter_BFER<>(*m.monitor)));
    // report the simulation throughputs
    u.reporters.push_back(std::unique_ptr<spu::tools::Reporter>(new tools::Reporter_throughput<>(*m.monitor)));
    // create a terminal that will display the collected data from the reporters
    u.terminal = std::unique_ptr<spu::tools::Terminal_std>(new spu::tools::Terminal_std(u.reporters));
}
