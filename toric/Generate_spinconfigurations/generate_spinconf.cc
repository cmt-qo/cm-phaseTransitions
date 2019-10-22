#include "sampler.cc"
using namespace std;

//lattice length
const int L=8;

//number of expectation values to be calculated per spin
const int N_loops=3;

//which state in the ground state manifold (possible choices: 0,1,2,3)
const int gs=0;

//maximum (absolute) field value in randomly distributed field configurations
double maxbeta=1.0;

//number of spins
double nspins_= 2*L*L;

//one generator per thread (define generator in loop)
//here: distribrutions for randomness
uniform_real_distribution<> distu_;
uniform_real_distribution<> distu2_;
uniform_int_distribution<> distnx_(0,L-1);
uniform_int_distribution<> distny_(0,L-1);
uniform_int_distribution<> distspins_(0,2*L*L-1);
uniform_int_distribution<> distperc_(0,99);
uniform_int_distribution<> distsign_(0,1);
random_device rdweights_;


//functions to generate field configurations

void make_weights_nthspin(vector<double> & spinweights, int spin_i){
   for (int i=0; i<(2*L*L); i++){
     if (i==spin_i){
       spinweights[i]=1.0;}
     else{
       spinweights[i]=0;}}
   }

void make_weights_I2D(vector<double> & spinweights){
   //modifies spinweights, returns configuration with spinweights[i]=1 for all i
   for (int i=0; i<(2*L*L);i++){
          spinweights[i]=1.0;}
   }


vector<double> add_rf_to_category(vector<double> spinweights, int spin_i, double beta, double maxf, mt19937 & genweights_, double probsign=100){
     for (int i=0; i<(2*L*L);i++){
        double rfactor=1.0;
        int rdense=distperc_(genweights_);
        if(rdense>=probsign)
          rfactor=0.05;
        double rweight=distu2_(genweights_);
        rweight=rweight*maxf*rfactor;
        double rsign=distsign_(genweights_);
        if (i==spin_i){
          spinweights[i]=spinweights[i]*beta*maxbeta;
          }
        if (i!=spin_i){
          if (rsign==0)
             spinweights[i]=(spinweights[i]+rweight)*maxbeta;
          else if (rsign==1)
             spinweights[i]=(spinweights[i]-rweight)*maxbeta;
          }
        }
     return spinweights;
     }


vector<double> add_random_field_I2D(vector<double> spinweights, double randfield, mt19937 & genweights_){
   for (int i=0; i<(2*L*L);i++){
       double rweight=distu2_(genweights_);
       double rsign=distsign_(genweights_);
       if (rsign==0)
          spinweights[i]=spinweights[i]+randfield*rweight;
       else if (rsign==1)
          spinweights[i]=spinweights[i]-randfield*rweight;
       }
   return spinweights;
   }



//MC: returns expectation value of X-loop specified by 'weights' using the Metropolis-Hastings algorithm 
//IN: 'weights: vector of length (2L^2,3), weights[i,loop]=1 if sigma_z^i is included in the X-loop that is measured, 0 otherwise
//spinweights: it is measured on ground state with field configuration 'spinweights' 
//measured every messung*sweeps after thermalization time 'thermalize'
//beta: field strength
//index: helper index
//IN/OUT: call-by-reference arguments 'wert1' and 'wert2', returning measured values along the Markov Chain to compute expectation value of X-loops
vector<int> MC( double beta,int thermalize, vector<double> spinweights, mt19937 & gen_){
    //spin lattice, Markov chain flips spins on the lattice
    vector<int> spins(2*L*L); 

    //magnetization
    double M=0;

    //introduce non-contractible loops on the initial spin-configuration depending on the ground state choice. Here, we train on gs=0: ground state without non-contractible loops
    bool horizontalXloop=false;
    bool verticalXloop=false;
    if (gs==1)
      horizontalXloop=true;
    if (gs==2)
      verticalXloop=true;
    if (gs==3){
      horizontalXloop=true;
      verticalXloop=true;}
    for (int i=0; i<2*L*L; i++){  
         spins[i]=1;  
         M+=spins[i];
          }
    if (horizontalXloop){
      for (int i=0; i<2*L; i++){  //non-contractible horizontal X-loop (in first row)
         if (i%2==0){
           spins[i]=-spins[i]; 
           M+=2*spins[i];}
         }
      }
    if (verticalXloop){
      for (int j=0; j<L; j++){  //non-contractible vertical X-loop (in first column)
        spins[2*L*j+1]=-spins[2*L*j+1];
        M+=2*spins[2*L*j+1];
        }
      }       
    
    unsigned int maxit=thermalize*L*L;
     
    
    for (int n=0; n<maxit; n++){
        
        //choose random vertex to be flipped
        int x;
        x=distnx_(gen_);//random vertex x-coordinate
        int y;
        y=distnx_(gen_);//random vertex y-Koordinate
        
        //calculate 'energy' difference when flipping the spins around the vertex specified by (x,y)
        double dE;
        int ym1,xp1;
        ym1=y-1;
        xp1=x+1;
        
        if (y==0){
           ym1=L-1;}
        if (x==(L-1)){
           xp1=0;}
        int l1D, o1D, r1D, u1D;
        
        l1D=get_index1D(x,ym1,1,L);
        o1D=get_index1D(x,y,0,L);
        r1D=get_index1D(x,y,1,L);
        u1D=get_index1D(xp1,y,0,L);
        
        dE=2.0*beta*(spins[l1D]*spinweights[l1D]+spins[o1D]*spinweights[o1D]+spins[r1D]*spinweights[r1D]+spins[u1D]*spinweights[u1D]);  
        //cout<<dE<<" ";
        double rnumber;
        rnumber=distu_(gen_);
        //acceptance of vertex flip (Metropolis-Hastings test)
        //acceptance of vertex flip
        if (rnumber<=exp(-dE)){
           spins[l1D]=(-1)*spins[l1D];
           spins[o1D]=(-1)*spins[o1D];
           spins[r1D]=(-1)*spins[r1D];
           spins[u1D]=(-1)*spins[u1D];
           M=M+2*(spins[l1D]+spins[o1D]+spins[r1D]+spins[u1D]);
           }
        }
    return spins;
    }



int main(){

//field configuration
int conf=6;

//useful quantities for computation (randomness)
double rdweights=rdweights_();
random_device rd_seed;
double seed_=rd_seed();

//define field strengths of spin i, on which we measure expectation values and train, save field values as 'betas' 
const int Tval=100;
double betas[Tval];
for (int i=0;i<100;i++){
   if (i==0){
       betas[i]=0.0;
       }
   else{
       betas[i]=betas[i-1]+0.01;
       }
   }
  
//parameters for Monte Carlo sampling
unsigned int sweeps=500000;
int messung=20;
int thermalize=0.1*sweeps;

//can increase NumMC for more training examples 
const int NumMC=1;  
for (int stateMC_i=0; stateMC_i<NumMC; stateMC_i++){
   //new variable to store field configuration in
   vector<double> spinweights(2*L*L);
   vector<vector<double> > fieldconf(Tval,vector<double>(2*L*L));

   for (int i=0; i<(2*L*L);i++){
    spinweights[i]=0;
    }


   //###############################################################
   make_weights_I2D(spinweights);
   if (conf==1){
     for (int i=0; i<(2*L*L);i++){
       if (i%2==0){
         spinweights[i]=0.5;}
       }
    }

   if (conf==2){
     for (int i=0; i<(2*L*L);i++){
       if (i%3==1){
         spinweights[i]=0.5;}
       if (i%3==2){
         spinweights[i]=0.25;}
       }
      }

   if (conf==3){
    for (int i=0; i<(2*L*L);i++){
      if (i%10==9){
       spinweights[i]=-0.5;}
      }
   }


   if (conf==4){
     for (int i=0; i<(2*L*L);i++){
      if (i%20==19){
       spinweights[i]=-0.5;}
      if (i%5==4){
       spinweights[i]=0.25;}
      }
    }

   if (conf==5){
     for (int i=0; i<(2*L*L);i++){
       if (i%5==0){
         spinweights[i]=0.0;}
       }
     }

   //#################################################################
   
   ofstream file_b;
   ostringstream fileNameStream_b("");
   fileNameStream_b<<"betas.txt";
   string fileName_b=fileNameStream_b.str();
   file_b.open(fileName_b.c_str());

   

   for(int i=0; i<Tval; i++){
      file_b<<betas[i]<<" ";}
   file_b<<endl;
   file_b<<endl;

   file_b.close();

   //store projected configurations: training/prediction data
   ofstream file_a;
   ostringstream fileNameStream_a("");
   fileNameStream_a<<"spinconfs.txt";
   string fileName_a=fileNameStream_a.str();
   file_a.open(fileName_a.c_str());


   
   //store expectation values of stabilizers in 'expvalues'
   vector<vector<double> > spinconf(Tval,vector<double>(2*L*L)); //array fuer expvalues von loop fuer betas
   
   
   //------------------------------------------------------------------------------------------------------------
   //Monte Carlo: sampling
   #pragma omp parallel for num_threads(4)   
   for (int i=0; i<Tval; i++){
          //cout<<i<<endl;
          //random number generators (inside parallelized loop, such that there is one generator for each thread)
          //random numbers for MC sampling
          mt19937 gen_;
          Seed(gen_);   

          //random numbers to generate fieldconfigurations 
          mt19937 genweights_;
          genweights_.seed(stateMC_i*Tval+i+rdweights+seed_);

          double beta_=betas[i];
          //##############################################################################
          //create random field configuration with field strength beta_ on spin 'spini', random fields with magnitude smaller than maxbeta on all other fields  
          vector<double> spinweights2=spinweights;
          for(int spinweights_i=0; spinweights_i<2*L*L; spinweights_i++){
             spinweights2[spinweights_i]=spinweights2[spinweights_i]*beta_;}
          //##############################################################################  
          

          //helper variable
          int index=0;

          //run sampling process via Metropolis-Hastings algorithm
          vector<int> spins=MC(1.0,thermalize, spinweights2, gen_);    
          for (int sc_i=0; sc_i<2*L*L; sc_i++){
             spinconf[i][sc_i]=spins[sc_i];}
          
          }

   //write sampled configurations (training data) into file
   for(int loop3_i=0; loop3_i<Tval;loop3_i++){
     for(int i=0; i<2*L*L; i++){
        file_a<<spinconf[loop3_i][i]<<" ";}
     file_a<<endl;}
     

   file_a.close();
   //MC sampling up to here  
   //----------------------------------------------------------------------------------
   }

              
 return 0;}

