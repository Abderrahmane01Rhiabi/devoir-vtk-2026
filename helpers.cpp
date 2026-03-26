#include "helpers.h"

TimingsManager   *timer = TimingsManager::Initialize("");

const char *helpers_prefix = "";

void
GetMemorySize(const char *description)
{
    //unsigned int size = 0;
    float size = 0.0;
 //   unsigned int rss  = 0;
    long rss  = 0;
    //m
    //melin
 unsigned int share = 0;
    unsigned int trs  = 0;
 unsigned int drs = 0;
    unsigned int lrs  = 0; 
unsigned int dt = 0;
//end melin 

#if defined(__APPLE__)
    struct mstats m = mstats();
//    size = (unsigned int)m.bytes_used; // The bytes used out of the bytes_total.
    size = (float)m.bytes_used; // The bytes used out of the bytes_total.
    
    rss = m.bytes_total; // not quite accurate but this should be the total
    //string unit="o";
                                   // amount allocated by malloc.
    float fact=size;
    string unit="o";
    
    
    
	 cerr  << helpers_prefix << "Memory: for \"" << description << "\", the program has requested virtual mem " <<fact<<  unit << "\", resident set " <<rss<<  unit<< endl;
	
	
    
    
    
#elif !defined(_WIN32)
    using std::ios_base;
    using std::ifstream;
    using std::string;
    
    double vm_usage     = 0.0;
    double resident_set = 0.0;
    
    // 'file' stat seems to give the most reliable results
    //
    ifstream stat_stream("/proc/self/stat",ios_base::in);
    
    // dummy vars for leading entries in stat that we don't care about
    //
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    string O, itrealvalue, starttime;
    
    // the two fields we want
    //
    unsigned long vsize;
    //long rss;
    
    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
    >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
    >> utime >> stime >> cutime >> cstime >> priority >> nice
    >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
    
    stat_stream.close();
    
    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage     = vsize / 1024.0;
    resident_set = rss * page_size_kb;
string unit="Ko";

    //fclose(file);
	cerr  << helpers_prefix << "Memory: for \"" << description << "\", the program has requested virtual mem " <<vm_usage<<  unit << "\", resident set " <<resident_set<<  unit<< endl;
	

#endif




}


// ************************************************************************* //
//                              TimingsManager.h                             //
// ************************************************************************* //

// useful macro for computing time of arrival at a particular line of code
#define DELTA_TOA_THIS_LINE TimingsManager::TimeSinceLine(__FILE__, __LINE__)
#define TOA_THIS_LINE TimingsManager::TimeSinceInit()




#ifndef HAVE_FTIME_PROTOTYPE
extern "C" {
int ftime(struct timeb *);
}
#endif



static int        firstTimer = -1;

static struct TIMEINFO initTimeInfo;


static void
GetCurrentTimeInfo(struct TIMEINFO &timeInfo)
{
#if defined(_WIN32)
    _ftime(&timeInfo);
#else
    gettimeofday(&timeInfo, 0);
#endif
}


double
TimingsManager::TimeSinceLine(const char *file, int line)
{
    static std::map<std::string, TIMEINFO> keyMap;

    // a way to re-initilize this static function's static member if
    // ever needed
    if ((file == 0) && (line < 0))
    {
        keyMap.clear();
        return 0.0;
    }

    struct TIMEINFO currentTime;
    GetCurrentTimeInfo(currentTime);

    char key[256];
    snprintf(key, sizeof(key), "%s#%d", file, line);
    if (keyMap.find(key) == keyMap.end())
    {
        keyMap[key] = currentTime;
        return DBL_MAX;
    }
    else
    {
        struct TIMEINFO lastTime = keyMap[key];
        keyMap[key] = currentTime;
        return DiffTime(lastTime, currentTime);
    }
}



double
TimingsManager::TimeSinceInit()
{
    struct TIMEINFO currentTimeInfo;
    GetCurrentTimeInfo(currentTimeInfo);
    return DiffTime(initTimeInfo, currentTimeInfo);
}



TimingsManager::TimingsManager()
{
    filename          = ".timings";
    openedFile        = false;
    numCurrentTimings = 0;
    enabled           = false;
    withholdOutput    = false;
    neverOutput       = false;
    outputAllTimings  = false;
}




TimingsManager *
TimingsManager::Initialize(const char *fname)
{
    if (timer != NULL)
        return timer;

    timer = new SystemTimingsManager;
    GetCurrentTimeInfo(initTimeInfo);

    timer->SetFilename(fname);
    timer->Enable();

    return timer;
}


void
TimingsManager::Finalize()
{
    if (timer)
    {
        if (firstTimer >= 0)
            timer->StopTimer(firstTimer, "Total component run time");
        timer->StopAllUnstoppedTimers();
        timer->WithholdOutput(false);
        timer->DumpTimings();
        delete timer;
        timer = 0;
    }
}



void
TimingsManager::SetFilename(const std::string &fname)
{
    if (fname == "")
        return;

    //
    // Make sure that the filename includes the whole path so all .timings
    // files will be written to the right directory.
    //
#if defined(_WIN32)
    if (!(fname[0] == 'C' && fname[1] == ':'))
#else
    if (fname[0] != '/')
#endif
    {
        char currentDir[1024];
#if defined(_WIN32)
        _getcwd(currentDir,1023);
#else
        getcwd(currentDir,1023);
#endif
        currentDir[1023]='\0';
        std::string filenameTmp(currentDir);
        if(filenameTmp[filenameTmp.size()-1] != '/')
            filenameTmp += "/";
        filename = filenameTmp + fname + ".timings"; 
    }
    else
    {
        filename = fname + ".timings";
    }
}


void
TimingsManager::Enable(void)
{
    enabled = true;
    if (firstTimer < 0)
        firstTimer = timer->StartTimer();
}



void
TimingsManager::Disable(void)
{
    enabled = false;
}



void
TimingsManager::WithholdOutput(bool v)
{
    withholdOutput = v;
}

void
TimingsManager::NeverOutput(bool v)
{
    neverOutput = v;
}



void
TimingsManager::OutputAllTimings(void)
{
    outputAllTimings = true;
    DumpTimings();
    outputAllTimings = false;
}



int
TimingsManager::FindFirstUnusedEntry(void)
{
    for (unsigned int i = 0 ; i < usedEntry.size() ; i++)
        if (!usedEntry[i])
            return i;

    return -1;
}



int
TimingsManager::StartTimer(bool forced)
{
    if (!enabled && !forced)
        return -1;
    numCurrentTimings += 1;
    int rv = PlatformStartTimer();
    if (rv == usedEntry.size())
    {
        usedEntry.push_back(true);
    }
    else if (rv > usedEntry.size())
    {
        cerr << "TimingsManager::StartTimer: Cannot start timer. "
               << "Returning -1 as if timing was disabled." << std::endl;
        return -1;
    }
    else
    {
        usedEntry[rv] = true;
    }

    return rv;
}



double
TimingsManager::StopTimer(int index, const std::string &summary, bool forced)
{
    double t = 0.;

    if (enabled || forced)
    {
        if (index >= 0 && index < usedEntry.size())
            usedEntry[index] = false;
        t = PlatformStopTimer(index);
        if (!neverOutput)
            times.push_back(t);
        numCurrentTimings -= 1;
        if (enabled && !neverOutput)
        {
            char msg[2048];
            snprintf(msg, 2048, "Timing: %s", summary.c_str());
            summaries.push_back(msg);
        }
    }

    DumpTimings(cerr);

    return t;
}



double
TimingsManager::LookupTimer(const std::string &nm)
{
    double val = 0.0;

    if (enabled)
    {
        int numT = times.size();
        cerr<<"numT= "<<numT<<endl;
        for (int i = 0 ; i < numT ; i++)
        {
            cerr<<i<<": "<<summaries[i]<<endl;
            if (summaries[i].find(nm,0) != std::string::npos)
            {
                val += times[i];
            }
        }
    }

    return val;
}

void
TimingsManager::DumpTimings(void)
{
    //
    // Return if timings disabled.
    //
    if (!enabled)
        return;
    if (withholdOutput && !outputAllTimings)
        return;
    if (neverOutput)
        return;
    if (filename == "")
    {
        cerr << "Attempted to DumpTimings without setting name of file" << endl; 
        return;
    }

    ofstream ofile;

    if (!openedFile)
    {
        //
        // We haven't opened up the file before, so blow it away from previous
        // times the program was run.
        //
        ofile.open(filename.c_str());
        openedFile = true;
    }
    else
    {
        //
        // We have already dumped the timings once while running this program,
        // so just append to that file.
        //
        ofile.open(filename.c_str(), ios::app);
    }

    if (ofile.fail())
    {
        cerr << "Unable to open file " << filename.c_str()
               << " to dump timings information." << endl;
    }
    else
    {
        ofile<<std::fixed;
        DumpTimings(ofile);
        ofile.close();
    }
}


void
TimingsManager::StopAllUnstoppedTimers()
{
    //
    // Stop all un-stopped timers
    //
    for (int i = 0 ; i < timer->GetNValues(); i++)
        if (usedEntry[i])
            timer->StopTimer(i, "A StartTimer call was unmatched!  Elapsed time since StartTimer: ");
}


void
TimingsManager::DumpTimings(ostream &out)
{
    //
    // Return if timings disabled.
    //
    if (!enabled)
    {
        return;
    }
    if (withholdOutput && !outputAllTimings)
        return;
    if (neverOutput)
        return;

    int numT = times.size();
    for (int i = 0 ; i < numT ; i++)
    {
        out << helpers_prefix << summaries[i].c_str() << " took " << times[i] << endl;
    }

    //
    // The next time we dump timings, don't use these values.
    //
    times.clear();
    summaries.clear();
}


double
TimingsManager::DiffTime(const struct TIMEINFO &startTime,
                         const struct TIMEINFO &endTime)
{
#if defined(_WIN32)
    // 
    // Figure out how many milliseconds between start and end times 
    //
    int ms = (int) difftime(endTime.time, startTime.time);
    if (ms == 0)
    {
        ms = endTime.millitm - startTime.millitm;
    }
    else
    {
        ms =  ((ms - 1) * 1000);
        ms += (1000 - startTime.millitm) + endTime.millitm;
    }

    return (ms/1000.);
#else
    double seconds = double(endTime.tv_sec - startTime.tv_sec) + 
                     double(endTime.tv_usec - startTime.tv_usec) / 1000000.;
                     
    return seconds;
#endif
}




int
SystemTimingsManager::PlatformStartTimer(void)
{
    struct TIMEINFO t;
    GetCurrentTimeInfo(t);
    int idx = FindFirstUnusedEntry();
    if (idx >= 0)
        values[idx] = t;
    else
    {
        values.push_back(t);
        idx = values.size()-1;
    }

    return idx;
}



double
SystemTimingsManager::PlatformStopTimer(int index)
{
    if (index < 0 || index >= values.size())
    {
        cerr << "Invalid timing index (" << index << ") specified." << endl;
        return 0.0;
    }

    struct TIMEINFO endTime;
    GetCurrentTimeInfo(endTime);
    double rv = DiffTime(values[index], endTime);
    return rv;
}

