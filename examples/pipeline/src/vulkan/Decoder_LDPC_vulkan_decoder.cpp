/*
sp_vulkan::Vulkan_decoder's non-virtual parts: the factory that picks between the two
implementations, and the process-wide default it picks by.

Both implementations are always compiled (there is no build-time switch any more, and no reason
for one: they differ only in host-side launch cost, and each is a couple of hundred lines). The
choice is made once at startup -- from --dec-vk-chain in main_gpu.cpp, or failing that from
SPU_LDPC_VULKAN_CHAIN -- and every decoder built afterwards, including the ones replicated stages
create in clone(), uses it.
*/

#ifdef DECODER_VULKAN

#include <cstdlib>
#include <cctype>
#include <string>

#include "Vulkan/Decoder_LDPC_vulkan_kernel.hpp"

namespace
{
// Read once, on the first call, so that a run cannot change implementation halfway through.
sp_vulkan::decoder_impl env_default_impl()
{
    const char* raw = std::getenv("SPU_LDPC_VULKAN_CHAIN");
    if (raw == nullptr) return sp_vulkan::decoder_impl::CHAIN_CACHED;

    sp_vulkan::decoder_impl impl = sp_vulkan::decoder_impl::CHAIN_CACHED;
    sp_vulkan::Vulkan_decoder::str_to_impl(raw, impl); // an unknown value keeps the default
    return impl;
}

sp_vulkan::decoder_impl& default_impl_storage()
{
    static sp_vulkan::decoder_impl impl = env_default_impl();
    return impl;
}

}

namespace sp_vulkan
{

Vulkan_decoder*
Vulkan_decoder::create(int device_id, decoder_impl impl)
{
    switch (impl)
    {
        case decoder_impl::CHAIN_REBUILT: return make_decoder_chain_rebuilt(device_id);
        case decoder_impl::CHAIN_CACHED:
        default:                          return make_decoder_chain_cached(device_id);
    }
}

Vulkan_decoder*
Vulkan_decoder::create(int device_id)
{
    return create(device_id, get_default_impl());
}

void
Vulkan_decoder::set_default_impl(decoder_impl impl)
{
    default_impl_storage() = impl;
}

decoder_impl
Vulkan_decoder::get_default_impl()
{
    return default_impl_storage();
}

bool
Vulkan_decoder::str_to_impl(const std::string& s, decoder_impl& out)
{
    std::string lowered;
    lowered.reserve(s.size());
    for (char c : s) lowered.push_back((char)std::tolower((unsigned char)c));

    if (lowered == "cached" || lowered == "1")
    {
        out = decoder_impl::CHAIN_CACHED;
        return true;
    }
    if (lowered == "rebuilt" || lowered == "rebuild" || lowered == "0")
    {
        out = decoder_impl::CHAIN_REBUILT;
        return true;
    }
    return false;
}

const char*
Vulkan_decoder::impl_to_str(decoder_impl impl)
{
    return impl == decoder_impl::CHAIN_CACHED ? "cached" : "rebuilt";
}

}

#endif // DECODER_VULKAN
