// UCTF Readout Methodology launcher: unpacks the embedded page and opens it in an app window.
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;

class Launcher
{
    [STAThread]
    static void Main()
    {
        string dir = Path.Combine(Path.GetTempPath(), "uctf_readout");
        Directory.CreateDirectory(dir);
        string page = Path.Combine(dir, "index.html");
        using (Stream src = Assembly.GetExecutingAssembly().GetManifestResourceStream("index.html"))
        using (FileStream dst = File.Create(page))
            src.CopyTo(dst);
        string url = new Uri(page).AbsoluteUri;
        string[] browsers = {
            Environment.ExpandEnvironmentVariables(@"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
            Environment.ExpandEnvironmentVariables(@"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
            Environment.ExpandEnvironmentVariables(@"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            Environment.ExpandEnvironmentVariables(@"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        };
        foreach (string b in browsers)
            if (File.Exists(b)) { Process.Start(b, "--app=\"" + url + "\" --window-size=1600,900"); return; }
        Process.Start(url);   // fall back to the default browser
    }
}
