int px,py,x,r = 10;
float gravity = 0.5;
float increase = 15;
int highscore = 0;
int frameincrease = 50;
GeneticAlgorithm ga = new GeneticAlgorithm(100);
void setup(){
  frameRate(60);
  size(500,500,P2D);
  x = width/4;
  px = width;
  py = round(random(100,height-100));
}
void draw(){
  run();
}
void run(){
  background(0);
  stroke(255);
  line(0,0,width,0);
  line(0,0,0,height);
  line(width-1,0,width-1,height-1);
  line(0,height-1,width-1,height-1);
  noStroke();
  text("Highscore:"+highscore,10,20);
  Pillar();
  if(ga.agents.size() > 0){
    ga.update();
    ga.display();
  }else{
    ga.nextgeneration();
    px = width;
    py = round(random(100,height-100));
  }
}
void Pillar(){
  px--;
  if(px < 0){
    px = width;
    py = round(random(100,height-100));
  }
  fill(255);
  noStroke();
  rect(px,0,2,py-(frameincrease+25));
  rect(px,height,2,py-height+(frameincrease+25));
}
void mousePressed(){
  frameRate(map(mouseX,0,width,1,1000));
}
class agent{
  int index;
  float y = height*0.5;
  float yvel = 0;
  int score = 0;
  double fitness;
  int framecount;
  Network brain;
  agent(int index_){
    index = index_;
    brain = new Network(4,1,1,1);
  }
  void inputs(){
    brain.inputs[0] = (px-x)/width;
    brain.inputs[1] = (py-y)/height;
    brain.inputs[2] = yvel/15;
    brain.inputs[3] = 1-(framecount/frameincrease);
  }
  boolean update(){
    yvel+=gravity;
    yvel = constrain(yvel,-15,4);
    y+=yvel;
    if(y-r*0.5 < py-(frameincrease+25) && x-r*0.5 < px && x+r*0.5 > px){
      return(true);
    }
    if(y+r*0.5 > py+(frameincrease+25) && x-r*0.5 < px && x+r*0.5 > px){
      return(true);
    }
    if(y+r*0.5 > height){
      return(true);
    }
    if(y-r*0.5 < 0){
      return(true);
    }
    framecount--;
    framecount = constrain(framecount,0,frameincrease);
    inputs();
    brain.feedforward();
    if(brain.outputs[0] > 0){
      flap();
    }
    score++;
    if(score > highscore){
      highscore = score;
    }
    return(false);
  }
  void flap(){
    if(framecount < 1){
      yvel-=increase;
      framecount+=frameincrease;
    }
  }
  void display(){
    fill(255,140);
    ellipse(x,y,r,r);
  }
}
class GeneticAlgorithm{
  int population;
  int GenerationCount = 0;
  ArrayList<agent> agents;
  agent[] agentscopy;
  GeneticAlgorithm(int population_){
    population = population_;
    agents = new ArrayList<agent>();
    agentscopy = new agent[population];
    for(int index = 0; index < population; index++){
      agents.add(new agent(index));
    }
    for(int index = 0; index < population; index++){
      agent current = agents.get(index);
      agentscopy[index] = current;
    }
  }
  void update(){
    for(int index = 0; index < agents.size(); index++){
      agent current = agents.get(index);
      if(current.update()){
        agentscopy[current.index] = current;
        agents.remove(index);
      }
    }
  }
  void display(){
    for(int index = 0; index < agents.size(); index++){
      agent current = agents.get(index);
      current.display();
    }
  }
  void calculatefitness(){
    float sum = 0;
    for(int index = 0; index < population; index++){
      sum+=agentscopy[index].score;
    }
    for(int index = 0; index < population; index++){
      agentscopy[index].fitness = agentscopy[index].score/sum+(2/population);
    }
  }
  void nextgeneration(){
    GenerationCount++;
    calculatefitness();
    for(int index = 0; index < population; index++){
      agents.add(new agent(index));
      agent current = agents.get(index);
      current.brain = select().brain;
      current.brain.mutate();
    }
  }
  agent select() {
    int index = 0;
    float r = random(0.99998);
    while (r > 0) {
      r -= agentscopy[index].fitness;
      index++;
    }
    if(index > 0){
      index--;
    }
    return agentscopy[index];
  }
}
class Network{
  float[] outputs;
  float[][] inhl;
  float[][][] hlhl;
  float[][] hlout;
  float[][] hlVals;
  float[] inputs;
  int in,hl,out,l;
  Network(int in_, int hl_, int out_, int l_){
    outputs = new float[out_];
    inputs = new float[in_];
    in = in_;
    hl = hl_;
    out = out_;
    l = l_;
    inhl = new float[in][hl];
    hlhl = new float[l][hl][hl];
    hlout = new float[hl][out];
    hlVals = new float[l][hl];
    for(int i = 0; i < in; i++){
      for(int n = 0; n < hl; n++){
        inhl[i][n] = random(-1,1);
      }
    }
    for(int i = 0; i < l; i++){
      for(int n = 0; n < hl; n++){
        hlhl[i][n][n] = random(-1,1);
      }
    }
    for(int i = 0; i < hl; i++){
      for(int n = 0; n < out; n++){
        hlout[i][n] = random(-1,1);
      }
    }
  }
  void mutate(){
    for(int i = 0; i < in; i++){
      for(int n = 0; n < hl; n++){
        if(random(1) > 0.9){
          inhl[i][n]+=random(-1,1);
        }
      }
    }
    for(int i = 0; i < l; i++){
      for(int n = 0; n < hl; n++){
        for(int a = 0; a < hl; a++){
          if(random(1) > 0.9){
            hlhl[i][n][a]+=random(-1,1);
          }
        }
      }
    }
    for(int i = 0; hl < hl; i++){
      for(int n = 0; n < out; n++){
        if(random(1) > 0.9){
          hlout[i][n]+=random(-1,1);
        }
      }
    }
  }
  void feedforward(){
    for(int i = 0; i < hl; i++){
      float total = 0;
      for(int n = 0; n < in; n++){
        total+=inputs[n]*inhl[n][i];
      }
      total/=in;
      hlVals[0][i] = activate(total);
    }
    for(int i = 1; i < l; i++){
      for(int n = 0; n < hl; n++){
        float total = 0;
        for(int a = 0; a < hl; a++){
          total+=hlVals[i-1][n]*hlhl[i][n][a];
        }
        hlVals[i][n] = total;
      }
    }
    for(int i = 0; i < out; i++){
      float total = 0;
      for(int n = 0; n < hl; n++){
        total+=hlVals[l-1][n]*hlout[n][i];
      }
      total/=hl;
      outputs[i]=total;
    }
  }
}
float activate(float x){
  int add = 0;
  if(x >= 0){
    add = 1;
  }else{
    add = -1;
  }
  return(x/(x+add))*add;
}
