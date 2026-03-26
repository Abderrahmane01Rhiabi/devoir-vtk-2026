#include "exo-vtk-include.h"
#include "config.h"
#include "helpers.h"

#include <vtkAppendPolyData.h>
#include <vtkCleanPolyData.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkUnsignedCharArray.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif


// --- choix du fichier de donnees (decommenter UN bloc) ----------------------

/* // frog (petit, CHAR, pour tester - on doit voir une grenouille)
#define FICHIER  "Frog_CHAR_X_256_Y_256_Z_44.raw"
int gridSize  = 256;
int YgridSize = 256;
int ZgridSize = 44;
#define CHAR
#define SMALL
int startexploreval = 13;
int endexploreval   = 200;
*/


/* // mystere1 (moyen, SHORT)
#define FICHIER  "Mystere1_SHORT_X_512_Y_512_Z_134.raw"
int gridSize  = 512;
int YgridSize = 512;
int ZgridSize = 134;
#define SHORT
#define SMALL
int startexploreval = 100;
int endexploreval   = 65000;
*/


// mystere2 (moyen, SHORT)
#define FICHIER  "Mystere2_SHORT_X_512_Y_400_Z_512.raw"
int gridSize  = 512;
int YgridSize = 400;
int ZgridSize = 512;
#define SHORT
#define SMALL
int startexploreval = 100;
int endexploreval   = 65000;



/* // mystere5 (tres grand, SHORT, BIG)
#define FICHIER  "Mystere5_SHORT_X_2048_Y_2048_Z_756.raw"
int gridSize  = 2048;
int YgridSize = 2048;
int ZgridSize = 756;
#define SHORT
#define BIG
int startexploreval = 100;
int endexploreval   = 65000;
*/

/* // mystere6 (grand, CHAR, BIG)
#define FICHIER  "Mystere6_CHAR_X_1118_Y_2046_Z_694.raw"
int gridSize  = 1118;
int YgridSize = 2046;
int ZgridSize = 694;
#define CHAR
#define BIG
int startexploreval = 1;
int endexploreval   = 255;
*/

/* // mystere10 (CHAR, BIG)
#define FICHIER  "Mystere10_CHAR_X_1204_Y_1296_Z_224.raw"
int gridSize  = 1204;
int YgridSize = 1296;
int ZgridSize = 224;
#define CHAR
#define BIG
int startexploreval = 37;
int endexploreval   = 255;
*/

/* // mystere11 (SHORT, SMALL)
#define FICHIER  "Mystere11_SHORT_X_512_Y_512_Z_1024.raw"
int gridSize  = 512;
int YgridSize = 512;
int ZgridSize = 1024;
#define SHORT
#define SMALL
int startexploreval = 100;
int endexploreval   = 65000;
*/







// --- constantes pour out-of-core ------------------------------------------------

// taille d'un bloc en tranches Z. on ne charge jamais plus que ca a la fois
// exemple : mystere8 (2048x2048 CHAR), 1 tranche = 4 Mo
// donc avec BLOCK_SIZE=16 on utilise 64 Mo max par bloc
static const int BLOCK_SIZE = 16;

// taille de la fenetre de rendu
static const int WIN_SIZE = 800;

// nombre d'images generees en mode batch
static const int NB_EXPLORE_VALUES = 8;





// --- prototypes des fonctions -----------------------------------------------

// lit un sous-volume (tranches zStart a zEnd) depuis le fichier .raw
// retourne un vtkRectilinearGrid alloue (a Delete() apres usage)
vtkRectilinearGrid* ReadBlock(int zStart, int zEnd);

// calcule l'isosurface sur un seul bloc, retourne un vtkPolyData
vtkPolyData* ComputeIsosurfaceBlock(vtkRectilinearGrid* block,
                                     double isoValue,
                                     int zStart, int zGlobal);

// mode out-of-core sequentiel : traite les blocs un par un
vtkPolyData* OOC_Sequential(double isoValue);

// mode out-of-core + openmp : blocs traites en parallele
vtkPolyData* OOC_Parallel(double isoValue);

// sauvegarde une image png depuis une fenetre de rendu
void SavePNG(vtkRenderWindow* renwin, const std::string& filename);

// affiche une barre de progression dans le terminal
void PrintProgress(int done, int total, const std::string& label);

// --- lecture memoire residente (linux /proc/self/statm) -------------------

static long GetResidentSetKB()
{
    std::ifstream f("/proc/self/statm");
    long pages_total, pages_rss;
    f >> pages_total >> pages_rss;
    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;
    return pages_rss * page_size_kb;
}


// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[])
{
    // lecture des arguments de la ligne de commande
    bool modeBatch = false;   // --batch : genere les images sans fenetre
    bool modeBench = false;   // --bench : compare les 3 modes

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--batch") modeBatch = true;
        if (std::string(argv[i]) == "--bench") modeBench = true;
    }

    // infos de demarrage
    std::cerr << "============================================================\n";
    std::cerr << "  devoir visualisation vtk 2026\n";
    std::cerr << "  fichier  : " << FICHIER << "\n";
    std::cerr << "  dims     : " << gridSize << " x " << YgridSize
              << " x " << ZgridSize << "\n";
    std::cerr << "  bloc ooc : " << BLOCK_SIZE << " tranches Z\n";

#ifdef _OPENMP
    std::cerr << "  openmp   : " << omp_get_max_threads() << " threads\n";
#else
    std::cerr << "  openmp   : non disponible\n";
#endif

    std::cerr << "  mode     : "
              << (modeBatch ? "batch" : modeBench ? "bench" : "interactif")
              << "\n";
    std::cerr << "============================================================\n";

    GetMemorySize("demarrage");

    // creation de la fenetre vtk
    vtkRenderWindow* renwin = vtkRenderWindow::New();
    renwin->SetSize(WIN_SIZE, WIN_SIZE);
    // en mode batch/bench on rend hors ecran (pas de fenetre visible)
    if (modeBatch || modeBench) renwin->SetOffScreenRendering(1);

    vtkRenderer* ren = vtkRenderer::New();
    ren->SetBackground(0.1, 0.1, 0.2);  // fond bleu fonce
    renwin->AddRenderer(ren);

    // table de couleurs : degrade chaud
    vtkLookupTable* lut = vtkLookupTable::New();
    lut->SetHueRange(0.0, 0.15);
    lut->SetSaturationRange(0.3, 1.0);
    lut->SetValueRange(1.0, 1.0);
    lut->SetNumberOfColors(256);
    lut->Build();


    // =========================================================================
    // mode benchmark : compare les 3 approches sur une valeur fixe
    // =========================================================================
    if (modeBench)
    {
        double testIso = (startexploreval + endexploreval) / 2.0;
        std::cerr << "\n[bench] valeur isosurface de test : " << testIso << "\n\n";

        // --- mode 1 : chargement complet (comme le canevas de base) ----------
        {
            long memBefore = GetResidentSetKB();
            auto t0 = std::chrono::high_resolution_clock::now();

            vtkRectilinearGrid* full = ReadBlock(0, ZgridSize - 1);

            vtkContourFilter* cf = vtkContourFilter::New();
            cf->SetInputData(full);
            cf->SetNumberOfContours(1);
            cf->SetValue(0, testIso);
            cf->Update();

            auto t1 = std::chrono::high_resolution_clock::now();
            long memAfter = GetResidentSetKB();

            double secs = std::chrono::duration<double>(t1 - t0).count();
            long   mem  = memAfter - memBefore;

            std::cerr << "[mode 1 - naif complet]   temps=" << std::fixed
                      << std::setprecision(2) << secs << "s   RAM+"
                      << mem << " ko\n";

            cf->Delete();
            full->Delete();
        }

        GetMemorySize("apres mode naif");

        // --- mode 2 : out-of-core sequentiel ---------------------------------
        {
            long memBefore = GetResidentSetKB();
            auto t0 = std::chrono::high_resolution_clock::now();

            vtkPolyData* result = OOC_Sequential(testIso);

            auto t1 = std::chrono::high_resolution_clock::now();
            long memAfter = GetResidentSetKB();

            double secs = std::chrono::duration<double>(t1 - t0).count();
            long   mem  = memAfter - memBefore;

            std::cerr << "[mode 2 - ooc sequentiel] temps=" << std::fixed
                      << std::setprecision(2) << secs << "s   RAM+"
                      << mem << " ko   polygones="
                      << result->GetNumberOfPolys() << "\n";

            result->Delete();
        }

        GetMemorySize("apres ooc sequentiel");

        // --- mode 3 : out-of-core + openmp -----------------------------------
        {
            long memBefore = GetResidentSetKB();
            auto t0 = std::chrono::high_resolution_clock::now();

            vtkPolyData* result = OOC_Parallel(testIso);

            auto t1 = std::chrono::high_resolution_clock::now();
            long memAfter = GetResidentSetKB();

            double secs = std::chrono::duration<double>(t1 - t0).count();
            long   mem  = memAfter - memBefore;

            std::cerr << "[mode 3 - ooc+openmp]     temps=" << std::fixed
                      << std::setprecision(2) << secs << "s   RAM+"
                      << mem << " ko   polygones="
                      << result->GetNumberOfPolys() << "\n";

            result->Delete();
        }

        GetMemorySize("apres ooc+openmp");

        std::cerr << "\n[bench] termine.\n";

        lut->Delete();
        ren->Delete();
        renwin->Delete();
        return 0;
    }


    // =========================================================================
    // mode batch : genere NB_EXPLORE_VALUES images png automatiquement
    // =========================================================================
    if (modeBatch)
    {
        std::cerr << "\n[batch] generation de " << NB_EXPLORE_VALUES
                  << " images entre " << startexploreval
                  << " et " << endexploreval << "\n";

        // valeurs reparties uniformement dans l'intervalle [start, end]
        double step = (double)(endexploreval - startexploreval)
                      / (NB_EXPLORE_VALUES - 1);

        // generation sequentielle des images (parallelisme interne par bloc)
        for (int k = 0; k < NB_EXPLORE_VALUES; ++k)
        {
            double isoVal = startexploreval + k * step;

            // calcul ooc+parallele de l'isosurface pour cette valeur
            vtkPolyData* mesh = OOC_Parallel(isoVal);

            if (mesh->GetNumberOfPolys() == 0) {
                std::cerr << "[batch] val=" << (int)isoVal
                          << " -> aucun polygone, image ignoree\n";
                mesh->Delete();
                continue;
            }

            // chaque thread cree sa propre fenetre hors-ecran (thread-safe)
            vtkRenderWindow*  rw  = vtkRenderWindow::New();
            vtkRenderer*      r   = vtkRenderer::New();
            rw->SetSize(WIN_SIZE, WIN_SIZE);
            rw->SetOffScreenRendering(1);
            rw->AddRenderer(r);
            r->SetBackground(0.08, 0.08, 0.15);

            // normalisation : tous les volumes sont ramenes a la meme echelle
            // pour que l'image soit bien cadree quelle que soit la taille
            int maxDim = std::max(gridSize, std::max(YgridSize, ZgridSize));
            vtkSmartPointer<vtkTransform> tf = vtkSmartPointer<vtkTransform>::New();
            tf->Scale((double)gridSize  / maxDim,
                      (double)YgridSize / maxDim,
                      (double)ZgridSize / maxDim);
            vtkSmartPointer<vtkTransformFilter> tfFilter =
                vtkSmartPointer<vtkTransformFilter>::New();
            tfFilter->SetInputData(mesh);
            tfFilter->SetTransform(tf);

            vtkPolyDataMapper* mapper = vtkPolyDataMapper::New();
            mapper->SetInputConnection(tfFilter->GetOutputPort());
            mapper->SetScalarRange(startexploreval, endexploreval);
            mapper->SetLookupTable(lut);

            vtkActor* actor = vtkActor::New();
            actor->SetMapper(mapper);
            // un peu de transparence pour voir les structures internes
            actor->GetProperty()->SetOpacity(0.85);

            r->AddActor(actor);
            r->ResetCamera();
            r->GetActiveCamera()->Azimuth(20);
            r->GetActiveCamera()->Elevation(15);
            r->ResetCameraClippingRange();

            // nom du fichier png de sortie
            std::ostringstream fname;
            fname << "iso_" << std::string(FICHIER).substr(0, 10)
                  << "_val" << std::setw(6) << std::setfill('0')
                  << (int)isoVal << ".png";

                 // vue de face (deja la)
            SavePNG(rw, fname.str());

                 // vue 3/4
                 vtkCamera* cam1 = r->GetActiveCamera();
                 cam1->Azimuth(45);
                 cam1->Elevation(25);
                 r->ResetCameraClippingRange();
                 std::ostringstream fname2;
                 fname2 << "iso_" << std::string(FICHIER).substr(0, 10)
                     << "_val" << std::setw(6) << std::setfill('0')
                     << (int)isoVal << "_angle2.png";
                 SavePNG(rw, fname2.str());

                 // vue de dessus
                 cam1->Azimuth(0);
                 cam1->Elevation(90);
                 r->ResetCameraClippingRange();
                 std::ostringstream fname3;
                 fname3 << "iso_" << std::string(FICHIER).substr(0, 10)
                     << "_val" << std::setw(6) << std::setfill('0')
                     << (int)isoVal << "_angle3.png";
                 SavePNG(rw, fname3.str());

#ifdef _OPENMP
            #pragma omp critical
#endif
            {
                std::cerr << "[batch] val=" << std::setw(6) << (int)isoVal
                          << "  -> " << fname.str()
                          << "  (" << mesh->GetNumberOfPolys() << " polys)\n";
            }

            actor->Delete();
            mapper->Delete();
            r->Delete();
            rw->Delete();
            mesh->Delete();
        }

        std::cerr << "\n[batch] termine. images png dans le repertoire courant.\n";

        lut->Delete();
        ren->Delete();
        renwin->Delete();
        return 0;
    }


    // =========================================================================
    // mode interactif (par defaut) : fenetre vtk avec rotation a la souris
    // =========================================================================
    {
        std::cerr << "\n[interactif] calcul de l'isosurface (ooc+openmp)...\n";

        double isoVal = startexploreval;

        // mesure du temps de calcul
        auto t0 = std::chrono::high_resolution_clock::now();
        vtkPolyData* mesh = OOC_Parallel(isoVal);
        auto t1 = std::chrono::high_resolution_clock::now();

        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::cerr << "[interactif] calcul termine en " << std::fixed
                  << std::setprecision(2) << secs << "s   ("
                  << mesh->GetNumberOfPolys() << " polygones)\n";

        GetMemorySize("apres calcul isosurface");

        // normalisation spatiale
        int maxDim = std::max(gridSize, std::max(YgridSize, ZgridSize));
        vtkSmartPointer<vtkTransform> tf = vtkSmartPointer<vtkTransform>::New();
        tf->Scale((double)gridSize  / maxDim,
                  (double)YgridSize / maxDim,
                  (double)ZgridSize / maxDim);
        vtkSmartPointer<vtkTransformFilter> tfFilter =
            vtkSmartPointer<vtkTransformFilter>::New();
        tfFilter->SetInputData(mesh);
        tfFilter->SetTransform(tf);

        vtkPolyDataMapper* mapper = vtkPolyDataMapper::New();
        mapper->SetInputConnection(tfFilter->GetOutputPort());
        mapper->SetScalarRange(startexploreval, endexploreval);
        mapper->SetLookupTable(lut);

        vtkActor* actor = vtkActor::New();
        actor->SetMapper(mapper);

        ren->AddActor(actor);
        ren->ResetCamera();

        // interacteur vtk : rotation libre avec la souris (style trackball)
        vtkRenderWindowInteractor* iren = vtkRenderWindowInteractor::New();
        iren->SetRenderWindow(renwin);

        vtkInteractorStyleTrackballCamera* style =
            vtkInteractorStyleTrackballCamera::New();
        iren->SetInteractorStyle(style);

        renwin->Render();
        std::cerr << "[interactif] fenetre ouverte. fermer pour quitter.\n";
        iren->Start();

        style->Delete();
        iren->Delete();
        actor->Delete();
        mapper->Delete();
        mesh->Delete();
    }

    lut->Delete();
    ren->Delete();
    renwin->Delete();
    return 0;
}



// You should not need to modify these routines.

// =============================================================================
// ReadBlock - out-of-core : lit un sous-volume (tranches zStart a zEnd)
// =============================================================================
// c'est le coeur de l'out-of-core.
// au lieu de charger tout le fichier, on utilise seekg() pour se positionner
// directement a l'octet de la tranche zStart, et on ne lit que ce dont on
// a besoin.
//
vtkRectilinearGrid* ReadBlock(int zStart, int zEnd)
{
    int i;
    std::string file = MY_DATA_PATH + std::string(FICHIER);

    // ouverture en mode binaire
    std::ifstream ifile(file.c_str(), std::ios::binary);
    if (ifile.fail()) {
        std::cerr << "erreur : impossible d'ouvrir " << file << "\n";
        throw std::runtime_error("fichier introuvable : " + file);
    }

    int numSlices = zEnd - zStart + 1;

    // construction du vtkRectilinearGrid pour ce bloc.
    // un RectilinearGrid est une grille reguliere avec des coordonnees
    // separees selon X, Y, Z. on normalise dans [0,1] pour que le rendu
    // soit independant de la taille du volume.
    vtkRectilinearGrid* rg = vtkRectilinearGrid::New();

    // axe X : gridSize points uniformement dans [0,1]
    vtkFloatArray* X = vtkFloatArray::New();
    X->SetNumberOfTuples(gridSize);
    for (i = 0; i < gridSize; i++)
        X->SetTuple1(i, i / (gridSize - 1.0));
    rg->SetXCoordinates(X);
    X->Delete();

    // axe Y
    vtkFloatArray* Y = vtkFloatArray::New();
    Y->SetNumberOfTuples(YgridSize);
    for (i = 0; i < YgridSize; i++)
        Y->SetTuple1(i, i / (YgridSize - 1.0));
    rg->SetYCoordinates(Y);
    Y->Delete();

    // axe Z : seulement les tranches du bloc courant.
    // les coordonnees Z sont globales pour que les blocs se raccordent bien
    // quand on les fusionne avec vtkAppendPolyData.
    vtkFloatArray* Z = vtkFloatArray::New();
    Z->SetNumberOfTuples(numSlices);
    for (i = zStart; i <= zEnd; i++)
        Z->SetTuple1(i - zStart, i / (ZgridSize - 1.0));
    rg->SetZCoordinates(Z);
    Z->Delete();

    rg->SetDimensions(gridSize, YgridSize, numSlices);

    // nombre de scalaires par tranche (1 scalaire par point de grille)
    unsigned int valuesPerSlice = (unsigned int)gridSize * (unsigned int)YgridSize;

    // taille en octets d'une tranche selon le type de donnees
#if defined(SHORT)
    unsigned int bytesPerSlice = sizeof(unsigned short) * valuesPerSlice;
#elif defined(CHAR)
    unsigned int bytesPerSlice = sizeof(unsigned char)  * valuesPerSlice;
#elif defined(FLOAT)
    unsigned int bytesPerSlice = sizeof(float)          * valuesPerSlice;
#else
    #error "definissez SHORT, CHAR ou FLOAT"
#endif

    // offset en octets depuis le debut du fichier.
    // pour les gros fichiers (BIG) il faut des entiers 64 bits
#if defined(BIG)
    unsigned long long offset       = (unsigned long long)zStart * bytesPerSlice;
    unsigned long long bytesToRead  = (unsigned long long)bytesPerSlice * numSlices;
    unsigned long long valuesToRead = (unsigned long long)valuesPerSlice * numSlices;
#elif defined(SMALL)
    unsigned int offset       = (unsigned int)zStart * bytesPerSlice;
    unsigned int bytesToRead  = bytesPerSlice * numSlices;
    unsigned int valuesToRead = valuesPerSlice * numSlices;
#else
    #error "definissez BIG ou SMALL"
#endif

    // allocation du tableau vtk pour les scalaires.
    // on alloue exactement ce qu'il faut pour CE bloc seulement.
#if defined(SHORT)
    vtkUnsignedShortArray* scalars = vtkUnsignedShortArray::New();
    scalars->SetNumberOfTuples(valuesToRead);
    unsigned short* arr = scalars->GetPointer(0);
#elif defined(CHAR)
    vtkUnsignedCharArray* scalars = vtkUnsignedCharArray::New();
    scalars->SetNumberOfTuples(valuesToRead);
    unsigned char* arr = scalars->GetPointer(0);
#elif defined(FLOAT)
    vtkFloatArray* scalars = vtkFloatArray::New();
    scalars->SetNumberOfTuples(valuesToRead);
    float* arr = scalars->GetPointer(0);
#endif

    // lecture out-of-core : seekg() saute directement aux donnees du bloc.
    ifile.seekg(offset, std::ios::beg);
    ifile.read(reinterpret_cast<char*>(arr), bytesToRead);
    ifile.close();

    // calcul min/max pour aider a choisir les bonnes valeurs d'isosurface
#if defined(BIG)
    unsigned long long n = valuesToRead;
#else
    unsigned int n = valuesToRead;
#endif

    int vmin = 2147483647, vmax = 0;
    for (unsigned long long ii = 0; ii < (unsigned long long)n; ii++) {
        int v = (int)(scalars->GetPointer(0))[ii];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    // affichage seulement pour le premier bloc pour eviter le spam
    if (zStart == 0) {
        std::cerr << "    [readblock] min=" << vmin << "  max=" << vmax
                  << "  (bloc z=" << zStart << "..." << zEnd << ")\n";
    }

    // on attache les scalaires a la grille
    scalars->SetName("entropy");
    rg->GetPointData()->AddArray(scalars);
    rg->GetPointData()->SetActiveScalars("entropy");
    scalars->Delete();

    return rg;
}


// =============================================================================
// ComputeIsosurfaceBlock - calcule l'isosurface sur un seul bloc
// =============================================================================
// applique vtkContourFilter (marching cubes) sur le bloc.
// retourne un vtkPolyData avec les triangles de la surface iso.
//
vtkPolyData* ComputeIsosurfaceBlock(vtkRectilinearGrid* block,
                                     double isoValue,
                                     int /*zStart*/, int /*zGlobal*/)
{
    // vtkContourFilter : implementation de marching cubes dans vtk.
    // il genere un maillage triangulaire pour la valeur isoValue donnee.
    vtkContourFilter* cf = vtkContourFilter::New();
    cf->SetInputData(block);
    cf->SetNumberOfContours(1);
    cf->SetValue(0, isoValue);
    cf->Update();

    // on detache le resultat du pipeline pour pouvoir supprimer le filtre
    vtkPolyData* result = vtkPolyData::New();
    result->ShallowCopy(cf->GetOutput());

    cf->Delete();
    return result;
}


// =============================================================================
// OOC_Sequential - out-of-core sequentiel (sans parallelisme)
// =============================================================================
// traite les blocs un par un. a chaque iteration :
//   1. lit BLOCK_SIZE tranches (out-of-core avec seekg)
//   2. calcule l'isosurface sur ce bloc (marching cubes)
//   3. ajoute le resultat a l'accumulateur vtkAppendPolyData
//   4. libere la memoire du bloc -> la RAM ne contient qu'un seul bloc
//
vtkPolyData* OOC_Sequential(double isoValue)
{
    // vtkAppendPolyData fusionne plusieurs vtkPolyData en un seul maillage
    vtkAppendPolyData* appender = vtkAppendPolyData::New();

    // arrondi superieur pour le dernier bloc (qui peut etre plus petit)
    int numBlocks = (ZgridSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::cerr << "  [ooc-seq] " << numBlocks << " blocs de "
              << BLOCK_SIZE << " tranches\n";

    for (int b = 0; b < numBlocks; b++)
    {
        int zStart = b * BLOCK_SIZE;
        int zEnd   = std::min(zStart + BLOCK_SIZE - 1, ZgridSize - 1);

        PrintProgress(b, numBlocks, "ooc-seq");

        // etape 1 : lecture out-of-core (seekg vers zStart)
        vtkRectilinearGrid* block = ReadBlock(zStart, zEnd);

        // etape 2 : calcul de l'isosurface sur ce bloc seulement
        vtkPolyData* poly = ComputeIsosurfaceBlock(block, isoValue, zStart, ZgridSize);

        // etape 3 : ajout a l'accumulateur
        appender->AddInputData(poly);
        appender->Update();

        // etape 4 : liberation immediate de la memoire du bloc.
        block->Delete();
        poly->Delete();
    }
    PrintProgress(numBlocks, numBlocks, "ooc-seq");
    std::cerr << "\n";

    appender->Update();
    vtkPolyData* result = vtkPolyData::New();
    result->ShallowCopy(appender->GetOutput());
    appender->Delete();
    return result;
}


// =============================================================================
// OOC_Parallel - out-of-core + openmp
// =============================================================================
// meme principe que OOC_Sequential mais les blocs sont traites en parallele.
//
// vtk n'est pas thread-safe pour la creation d'objets, donc on separe en
// 2 phases :
//   phase 1 (parallele)    : chaque thread lit et calcule son bloc
//                            resultats stockes dans results[b]
//   phase 2 (sequentielle) : fusion dans vtkAppendPolyData
//
vtkPolyData* OOC_Parallel(double isoValue)
{
    int numBlocks = (ZgridSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

#ifdef _OPENMP
    int nThreads = omp_get_max_threads();
    std::cerr << "  [ooc-par] " << numBlocks << " blocs, "
              << nThreads << " threads openmp\n";
#else
    std::cerr << "  [ooc-par] openmp absent, mode sequentiel\n";
    return OOC_Sequential(isoValue);
#endif

    // tableau pour stocker le resultat de chaque bloc.
    std::vector<vtkPolyData*> results(numBlocks, nullptr);

    // --- phase 1 : calcul parallele ------------------------------------------
    // chaque thread ouvre le fichier de son cote independamment
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int b = 0; b < numBlocks; b++)
    {
        int zStart = b * BLOCK_SIZE;
        int zEnd   = std::min(zStart + BLOCK_SIZE - 1, ZgridSize - 1);

#ifdef _OPENMP
        #pragma omp critical
        { PrintProgress(b, numBlocks, "ooc-par"); }
#endif

        // lecture du bloc par ce thread
        vtkRectilinearGrid* block = ReadBlock(zStart, zEnd);

        // calcul marching cubes
        vtkPolyData* poly = ComputeIsosurfaceBlock(block, isoValue, zStart, ZgridSize);

        // stockage : pas de race condition car chaque thread ecrit a b different
        results[b] = poly;

        // liberation du bloc des que l'isosurface est calculee
        block->Delete();
    }
    std::cerr << "\n";

    // --- phase 2 : fusion sequentielle (vtk non thread-safe ici) -------------
    vtkAppendPolyData* appender = vtkAppendPolyData::New();
    for (int b = 0; b < numBlocks; b++) {
        if (results[b] && results[b]->GetNumberOfPolys() > 0) {
            appender->AddInputData(results[b]);
        }
    }
    appender->Update();

    vtkPolyData* result = vtkPolyData::New();
    result->ShallowCopy(appender->GetOutput());

    appender->Delete();
    for (int b = 0; b < numBlocks; b++)
        if (results[b]) results[b]->Delete();

    return result;
}


// =============================================================================
// SavePNG - capture et sauvegarde l'image de rendu en png
// =============================================================================
void SavePNG(vtkRenderWindow* renwin, const std::string& filename)
{
    renwin->Render();

    // vtkWindowToImageFilter capture le contenu de la fenetre de rendu
    vtkWindowToImageFilter* w2i = vtkWindowToImageFilter::New();
    w2i->SetInput(renwin);
    w2i->Update();

    vtkPNGWriter* writer = vtkPNGWriter::New();
    writer->SetInputConnection(w2i->GetOutputPort());
    writer->SetFileName(filename.c_str());
    writer->Write();

    w2i->Delete();
    writer->Delete();
}


// =============================================================================
// PrintProgress - barre de progression dans le terminal
// =============================================================================
void PrintProgress(int done, int total, const std::string& label)
{
    if (total <= 0) return;
    int pct   = (100 * done) / total;
    int width = 30;
    int fill  = (width * done) / total;

    std::cerr << "\r  [" << label << "] [";
    for (int i = 0; i < fill;  i++) std::cerr << "=";
    for (int i = fill; i < width; i++) std::cerr << " ";
    std::cerr << "] " << std::setw(3) << pct << "%  "
              << done << "/" << total << "  " << std::flush;
}