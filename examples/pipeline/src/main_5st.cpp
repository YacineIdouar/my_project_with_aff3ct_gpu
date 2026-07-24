#include <type_traits>
#include <functional>
#include <exception>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <random>

#include <aff3ct.hpp>
#include "Module/Decoder_gpu/Decoder_LDPC_BP_flooding_gpu.hpp"
#include "Module/Channel/Channel_AWGN_LLR_gpu.hpp"
using namespace aff3ct;

// ---------------------------------------------------------------------------
// Runtime GPU configuration.
//
// Which backend the decoder task runs on, and which device/platform every GPU module targets, used
// to be baked into the source: a literal '{VULKAN, 0, 0}' at the set_execution_device_info() call
// sites and a literal 0 in each module constructor. Both are runtime choices now, mirroring the
// per-task API selection of the motion project. Only backends actually compiled in (the DECODER_*
// CMake options) are accepted, so requesting one that was not built fails loudly rather than
// silently falling back to another.
// ---------------------------------------------------------------------------

static std::string compiled_decoder_apis()
{
    std::string list;
    auto add = [&list](const char* name) { list += list.empty() ? name : std::string(", ") + name; };
#ifdef DECODER_CUDA
    add("CUDA");
#endif
#ifdef DECODER_HIP
    add("HIP");
#endif
#ifdef DECODER_SYCL
    add("SYCL");
#endif
#ifdef DECODER_VULKAN
    add("VULKAN");
#endif
    return list.empty() ? std::string("<none>") : list;
}

// VULKAN is tried first so that the default matches the backend this main was previously pinned to.
static std::string default_decoder_api()
{
#if   defined(DECODER_VULKAN)
    return "VULKAN";
#elif defined(DECODER_CUDA)
    return "CUDA";
#elif defined(DECODER_HIP)
    return "HIP";
#elif defined(DECODER_SYCL)
    return "SYCL";
#else
    return "<none>";
#endif
}

static spu::device_interface::compute_api str_to_compute_api(const std::string& s, const char* opt)
{
#ifdef DECODER_CUDA
    if (s == "CUDA")   return spu::device_interface::compute_api::CUDA;
#endif
#ifdef DECODER_HIP
    if (s == "HIP")    return spu::device_interface::compute_api::HIP;
#endif
#ifdef DECODER_SYCL
    if (s == "SYCL")   return spu::device_interface::compute_api::SYCL;
#endif
#ifdef DECODER_VULKAN
    if (s == "VULKAN") return spu::device_interface::compute_api::VULKAN;
#endif
    std::cerr << "(EE) unsupported '" << opt << "' value '" << s << "' (compiled backends: "
              << compiled_decoder_apis() << ")" << std::endl;
    std::exit(1);
}

// The channel module exists in two unrelated flavours: the AFF3CT factory one (CPU) and the CUDA one
// declared in this example. They share their socket names but not their task name, so the choice has
// to be carried around at bind time.
enum class channel_impl { CPU, CUDA };

static const char* channel_task_name(const channel_impl impl)
{
    return impl == channel_impl::CUDA ? "add_noise_gpu" : "add_noise";
}

static channel_impl str_to_channel_impl(const std::string& s, const char* opt)
{
    if (s == "CPU")  return channel_impl::CPU;
    if (s == "CUDA") return channel_impl::CUDA;

    std::cerr << "(EE) unsupported '" << opt << "' value '" << s << "' (expected CPU or CUDA)"
              << std::endl;
    std::exit(1);
}

// Pulls "--opt <value>" out of argv, erasing both tokens. The AFF3CT factory parser only knows the
// factory arguments, so leaving ours in place would make it report them as unknown.
static bool extract_option(std::vector<char*>& args, const char* opt, std::string& out)
{
    for (size_t i = 1; i < args.size(); i++)
    {
        if (std::strcmp(args[i], opt) != 0) continue;
        if (i + 1 >= args.size())
        {
            std::cerr << "(EE) '" << opt << "' expects a value." << std::endl;
            std::exit(1);
        }
        out = args[i + 1];
        args.erase(args.begin() + i, args.begin() + i + 2);
        return true;
    }
    return false;
}

// These are consumed before the factory parser runs, so they never appear in its own help output.
static void print_gpu_options_help()
{
    std::cout << "GPU parameter(s):" << std::endl;
    std::cout << "  --dev-id  <int>   Device id used by every GPU module                     [0]"
              << std::endl;
    std::cout << "  --plt-id  <int>   Platform id (only SYCL indexes devices by platform)    [0]"
              << std::endl;
    std::cout << "  --dec-api <api>   Backend running the decoder task, one of {"
              << compiled_decoder_apis() << "}  [" << default_decoder_api() << "]" << std::endl;
    std::cout << "  --chn-api <api>   Channel implementation, one of {CPU, CUDA}             [CUDA]"
              << std::endl;
    std::cout << "  --snr-min  <flt>  First Eb/N0 (dB) of the sweep                          [1.00]"
              << std::endl;
    std::cout << "  --snr-max  <flt>  Sweep runs while Eb/N0 < this value (dB)               [4.01]"
              << std::endl;
    std::cout << "  --snr-step <flt>  Eb/N0 increment (dB) between points                    [0.25]"
              << std::endl;
    std::cout << std::endl;
}

static int extract_int_option(std::vector<char*>& args, const char* opt, const int def_value)
{
    std::string raw;
    if (!extract_option(args, opt, raw)) return def_value;

    try
    {
        size_t consumed = 0;
        const int value = std::stoi(raw, &consumed);
        if (consumed != raw.size() || value < 0) throw std::invalid_argument(raw);
        return value;
    }
    catch (const std::exception&)
    {
        std::cerr << "(EE) '" << opt << "' expects a non-negative integer (got '" << raw << "')."
                  << std::endl;
        std::exit(1);
    }
}

static float extract_float_option(std::vector<char*>& args, const char* opt, const float def_value)
{
    std::string raw;
    if (!extract_option(args, opt, raw)) return def_value;

    try
    {
        size_t consumed = 0;
        const float value = std::stof(raw, &consumed);
        if (consumed != raw.size()) throw std::invalid_argument(raw);
        return value;
    }
    catch (const std::exception&)
    {
        std::cerr << "(EE) '" << opt << "' expects a real number (got '" << raw << "')." << std::endl;
        std::exit(1);
    }
}

struct params
{
    size_t n_threads = std::thread::hardware_concurrency();
    float  ebn0      = 2.40f; // SNR value (single point, kept for reference)
    float  ebn0_min  = 0.00f; // --snr-min:  first Eb/N0 of the sweep
    float  ebn0_max  = 4.01f; // --snr-max:  loop runs while ebn0 < ebn0_max
    float  ebn0_step = 0.25f; // --snr-step: Eb/N0 increment between points
    float  R;                  // code rate (R=K/N)

    int dev_id      = 0;       // --dev-id:  GPU device id used by every GPU module
    int platform_id = 0;       // --plt-id:  GPU platform id (only SYCL indexes by platform)
    int dec_n_ite   = 10;      // --dec-ite: decoder iterations (read back from the codec factory)
    spu::device_interface::compute_api dec_api = // --dec-api: backend the decoder task executes on
        spu::device_interface::compute_api::NATIVE;
    channel_impl chn_api = channel_impl::CUDA; // --chn-api: CPU or CUDA channel module

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
    // Exactly one of the two channels below is built (see --chn-api); 'channel' aliases it and
    // 'channel_task' names its single task, both flavours exposing the same CP/X_N/Y_N sockets.
    std::unique_ptr<     module::Channel<>>                     channel_cpu;
	std::unique_ptr<     module::Channel_AWGN_LLR_gpu<float>>   channel_gpu;
                    spu::module::Module*                        channel = nullptr;
                         std::string                            channel_task;
    std::unique_ptr<     module::Monitor_BFER<>> monitor;
    std::unique_ptr<spu::module::Sink<>>         sink;
    std::unique_ptr<     tools ::Codec_SIHO<>>   codec;
                         module::Encoder<>*      encoder;
                         module::Decoder_LDPC_BP_flooding_gpu<float, int>* decoder;
						 //module::Decoder_SIHO<>* decoder;
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
	(*m.channel)[ m.channel_task + "::X_N" ] = (*m.modem  )[   "modulate::X_N2"     ];
    (*m.modem  )[  "demodulate::Y_N1"    ] = (*m.channel)[ m.channel_task + "::Y_N" ];
	(*m.puncturer)[ "depuncture::Y_N1"   ] = (*m.modem  )[ "demodulate::Y_N2"     ];
    (*m.decoder)[ "decode_siho_gpu::Y_N" ]  = (*m.puncturer)[ "depuncture::Y_N2"    ];
    (*m.monitor)["check_errors::U"       ] = (*m.source )[   "generate::out_data" ];
    (*m.monitor)["check_errors::V"       ] = (*m.decoder)["decode_siho_gpu::V_K"      ];
    (*m.sink   )[  "send_count::in_count"] = (*m.source )[   "generate::out_count"];
    (*m.sink   )[  "send_count::in_data" ] = (*m.decoder)["decode_siho_gpu::V_K"      ];
    std::vector<float> sigma(1);
    (*m.channel)[ m.channel_task + "::CP" ] = sigma;
    (*m.modem  )[  "demodulate::CP"      ] = sigma;

	// The GPU channel only ever registers a CUDA codelet, so its API is structural rather than a
	// choice; the CPU one has no device info to set at all. The decoder registers one codelet per
	// compiled backend, so --dec-api picks between them.
	if (p.chn_api == channel_impl::CUDA)
		(*m.channel)(m.channel_task).set_execution_device_info({spu::device_interface::compute_api::CUDA, p.dev_id, p.platform_id}, true);
	(*m.decoder)("decode_siho_gpu").set_execution_device_info({p.dec_api, p.dev_id, p.platform_id}, true);

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

    // display the BER/FER legend once, before sweeping the SNRs
    u.terminal->legend();

    // loop over the various SNRs (Eb/N0), simulating the BER/FER at each point
    for (auto ebn0 = p.ebn0_min; ebn0 < p.ebn0_max; ebn0 += p.ebn0_step)
    {
        // compute the current sigma for the channel noise
        const auto esn0 = tools::ebn0_to_esn0(ebn0, p.R, p.modem->bps);
        std::fill(sigma.begin(), sigma.end(), tools::esn0_to_sigma(esn0, p.modem->cpm_upf));

        // set_values fires the recorded callbacks, propagating the new noise to every module
        u.noise->set_values(sigma[0], ebn0, esn0);

        // display the performance (BER and FER) in real time (in a separate thread)
        u.terminal->start_temp_report();

        // run the pipeline until the monitor has gathered enough errors (is_done()) for this SNR,
        // the input file ends, or the user presses Ctrl+c
        u.pipeline->exec([&m]() -> bool
            {
                return m.monitor->is_done();
            });

        // display the performance (BER and FER) for this SNR in the terminal
        u.terminal->final_report();

        // reset the monitor error counters before moving on to the next SNR
        m.monitor->reset();

        // stop the whole sweep if the user requested an interruption (Ctrl+c); otherwise the
        // remaining SNRs would be skipped through instantly as exec() keeps returning on the signal
        if (spu::tools::Signal_handler::is_sigint())
            break;
    }

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
    // Consume our GPU options first, then hand what is left to the AFF3CT factory parser.
    std::vector<char*> args(argv, argv + argc);
    p.dev_id      = extract_int_option(args, "--dev-id", p.dev_id);
    p.platform_id = extract_int_option(args, "--plt-id", p.platform_id);

    std::string api_str = default_decoder_api();
    extract_option(args, "--dec-api", api_str);
    p.dec_api = str_to_compute_api(api_str, "--dec-api");

    std::string chn_str = "CUDA";
    extract_option(args, "--chn-api", chn_str);
    p.chn_api = str_to_channel_impl(chn_str, "--chn-api");

    p.ebn0_min  = extract_float_option(args, "--snr-min",  p.ebn0_min);
    p.ebn0_max  = extract_float_option(args, "--snr-max",  p.ebn0_max);
    p.ebn0_step = extract_float_option(args, "--snr-step", p.ebn0_step);
    if (p.ebn0_step <= 0.f)
    {
        std::cerr << "(EE) '--snr-step' must be strictly positive (got " << p.ebn0_step << ")."
                  << std::endl;
        std::exit(1);
    }

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
    tools::Command_parser cp((int)args.size(), args.data(), params_list, true);
    if (cp.parsing_failed())
    {
        cp.print_help    ();
        print_gpu_options_help();
        cp.print_warnings();
        cp.print_errors  ();
        std::exit(1);
    }

    std::cout << "# Simulation parameters: " << std::endl;
    tools::Header::print_parameters(params_list); // display the headers (= print the AFF3CT parameters on the screen)
    std::cout << "# * GPU ---------------------------------------------" << std::endl;
    std::cout << "#    ** Device id                     = " << p.dev_id      << std::endl;
    std::cout << "#    ** Platform id                   = " << p.platform_id << std::endl;
    std::cout << "#    ** Decoder API                   = " << api_str
              << " (compiled: " << compiled_decoder_apis() << ")" << std::endl;
    std::cout << "#    ** Channel API                   = " << chn_str << std::endl;
    std::cout << "#    ** SNR min  (Eb/N0, dB)          = " << p.ebn0_min  << std::endl;
    std::cout << "#    ** SNR max  (Eb/N0, dB)          = " << p.ebn0_max  << std::endl;
    std::cout << "#    ** SNR step (Eb/N0, dB)          = " << p.ebn0_step << std::endl;
    std::cout << "#" << std::endl;
    cp.print_warnings();

    // The GPU decoder is built by hand rather than through the codec factory, so take the iteration
    // count from the factory (i.e. from --dec-ite) instead of repeating a literal at its call site.
    if (auto* dec_ldpc = dynamic_cast<const factory::Decoder_LDPC*>(p.codec->dec.get()))
        p.dec_n_ite = dec_ldpc->n_ite;

     p.R = (float)p.codec->enc->K / (float)p.puncturer->N; // compute the code rate
}

void init_modules(const params &p, modules &m)
{
    m.source  = std::unique_ptr<spu::module::Source      <>>(p.source ->build());
    m.codec   = std::unique_ptr<     tools ::Codec_SIHO  <>>(p.codec  ->build());
    m.modem   = std::unique_ptr<     module::Modem       <>>(p.modem  ->build());
	if (p.chn_api == channel_impl::CUDA)
	{
		m.channel_gpu = std::unique_ptr<module::Channel_AWGN_LLR_gpu<float>>(new aff3ct::module::Channel_AWGN_LLR_gpu<float> (p.channel.get()->N, p.channel.get()->seed, p.dev_id, p.platform_id));
		m.channel     = m.channel_gpu.get();
	}
	else
	{
		m.channel_cpu = std::unique_ptr<module::Channel<>>(p.channel->build());
		m.channel     = m.channel_cpu.get();
	}
	m.channel_task = channel_task_name(p.chn_api);
    m.monitor = std::unique_ptr<     module::Monitor_BFER<>>(p.monitor->build());
    m.sink    = std::unique_ptr<spu::module::Sink        <>>(p.sink   ->build());
    m.encoder = &m.codec->get_encoder();
    m.decoder = new aff3ct::module::Decoder_LDPC_BP_flooding_gpu<float, int>(p.codec.get()->K, p.codec->enc.get()->N_cw, p.puncturer.get()->N, p.dec_n_ite, p.dev_id, p.platform_id);
	//m.decoder = &m.codec->get_decoder_siho();
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
            .add_last_task((*m.modem)("modulate"))  //                                      last  task of the stage
            .set_n_threads(4)) //          can run on a multiple threads (with replication)
        // ------------------------------------------------------------------------------------------------------------
        .configure_interstage_synchro(spu::tools::Pipeline_builder::Synchro_builder() // ---------- INTER-STAGE 1 <-> 2
            .set_buffer_size(3) //                                                          synchronization buffer size
            .set_active_waiting(false)) //                                          passive waiting for synchronization

		.add_stage(spu::tools::Pipeline_builder::Stage_builder() // ------------------------------------------- STAGE 1
            .add_first_task((*m.channel)(m.channel_task)) //                                          first task of the stage
            .add_last_task((*m.channel)(m.channel_task))  //                                      last  task of the stage
            .set_n_threads(2)) //          can run on a multiple threads (with replication)
        // ------------------------------------------------------------------------------------------------------------
        .configure_interstage_synchro(spu::tools::Pipeline_builder::Synchro_builder() // ---------- INTER-STAGE 1 <-> 2
            .set_buffer_size(3) //                                                          synchronization buffer size
            .set_active_waiting(false)) //                                          passive waiting for synchronization

		.add_stage(spu::tools::Pipeline_builder::Stage_builder() // ------------------------------------------- STAGE 1
           .add_first_task((*m.modem)("demodulate")) //                                          first task of the stage
           .add_last_task((*m.puncturer)("depuncture")) //                                      last  task of the stage
           .set_n_threads(1)) //          can run on a multiple threads (with replication)
       // // ------------------------------------------------------------------------------------------------------------
       	.configure_interstage_synchro(spu::tools::Pipeline_builder::Synchro_builder() // ---------- INTER-STAGE 1 <-> 2
       	    .set_buffer_size(3) //                                                          synchronization buffer size
       	    .set_active_waiting(false)) //                                          passive waiting for synchronization

		.add_stage(spu::tools::Pipeline_builder::Stage_builder() // ------------------------------------------- STAGE 1
           .add_first_task((*m.decoder)("decode_siho_gpu")) //                                          first task of the stage
           .add_last_task((*m.decoder)("decode_siho_gpu")) //                                      last  task of the stage
           .set_n_threads(2)) //          can run on a multiple threads (with replication)
       // // ------------------------------------------------------------------------------------------------------------
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

	// Setting the number of frames 
	//u.pipeline->set_n_frames(2);

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