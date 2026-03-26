#include <iostream>
#include <stdio.h>
#include <fstream>

#if defined(__APPLE__)
#include <malloc/malloc.h> // for mstat
#include <mach/mach.h>

#endif
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <pwd.h>

#include <ios>

#include <string>

using namespace std;


void GetMemorySize(const char *description);


// ************************************************************************* //
//                              TimingsManager.h                             //
// ************************************************************************* //

#include <sys/time.h>
#include <sys/timeb.h>

#include <string>
#include <vector>

#define TIMEINFO timeval


class TimingsManager
{
  public:
                               TimingsManager();
    virtual                   ~TimingsManager() {;};

    static double              TimeSinceLine(const char *file, int line);
    static double              TimeSinceInit();

    static TimingsManager     *Initialize(const char *);
    static void                Finalize();
    void                       SetFilename(const std::string &s);

    void                       Enable(void);
    void                       Disable(void);
    bool                       Enabled() const {return enabled;};

    int                        StartTimer(bool force=false);
    double                     StopTimer(int, const std::string &, bool force=false);
    double                     LookupTimer(const std::string &);

    static double              DiffTime();

    void                       DumpTimings(void);
    void                       DumpTimings(ostream &);

    void                       WithholdOutput(bool);
    void                       NeverOutput(bool);
    bool                       GetNeverOutputValue(void) {return neverOutput;};
    void                       OutputAllTimings();

  protected:
    std::string                filename;
    bool                       openedFile;
    int                        numCurrentTimings;
    bool                       enabled;
    bool                       withholdOutput;
    bool                       neverOutput;
    bool                       outputAllTimings;

    std::vector<double>        times;
    std::vector<std::string>   summaries;
    std::vector<bool>          usedEntry;

    static double              DiffTime(const struct TIMEINFO &startTime,
                                        const struct TIMEINFO &endTime);
    int                        FindFirstUnusedEntry(void);

    void                       StopAllUnstoppedTimers();

    virtual int                PlatformStartTimer(void) = 0;
    virtual double             PlatformStopTimer(int) = 0;
    virtual int                GetNValues() const = 0;
    virtual void               ClearValues() = 0;
};

class SystemTimingsManager : public TimingsManager
{
  public:
                               SystemTimingsManager() {;};
    virtual                   ~SystemTimingsManager() {;};
    void                       ClearValues()
                                   { values.clear(); };
    int                        GetNValues() const
                                   { return values.size(); };

  protected:
    std::vector<struct TIMEINFO> values;
    virtual int                PlatformStartTimer(void);
    virtual double             PlatformStopTimer(int);
};


extern TimingsManager   *timer ;

#include <map>
#include <float.h>
#include <time.h>
#include <stdio.h>

#include <unistd.h>
