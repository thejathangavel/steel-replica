import type { Metadata } from "next";
export const metadata: Metadata = { title: "SteelGenie" };
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{
        margin: 0,
        padding: 0,
        backgroundColor: "#0F172A",
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}>
        {children}
      </body>
    </html>
  );
}
