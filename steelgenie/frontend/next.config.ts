import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow rendering base64 data URIs as image src
  images: {
    dangerouslyAllowSVG: false,
  },
};

export default nextConfig;
