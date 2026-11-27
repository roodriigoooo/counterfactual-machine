export {};
type V = [number, number];
type Segment = {id:string;a:V;b:V;radius:number;material:{friction:number;restitution:number}};
type Scene = {schema_version:number;id:string;parent_id:string;family:string;seed:number;ball:{id:string;position:V;velocity:V;radius:number};bumper:{id:string;position:V;radius:number};segments:Segment[]};
type Goal = {ball_id:string;kind:string;region:{kind:string;center:V;radius:number;normal:V;offset:number};start:number;end:number};
type Trajectory = {times:number[];states:number[][];events:{time:number;phase:string;objects:string[]}[]};
type Outcome = {success:boolean;margin:number};
type Drop = {trajectory:Trajectory;goal:Outcome;simulator_calls?:number;new_simulator_calls?:number};
type Preset = {name:string;scene:Scene;goal:Goal};
type Plan = {plan_id:string;status:string;edit_m:number|null;predicted_margin_m:number|null;predicted_original:number[][];predicted_edited:number[][]|null;candidate_count:number;seconds:number;grid_m:number[];margins_m:(number|null)[]};
const $=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
const input=(id:string)=>$<HTMLInputElement>(id);
const dialog=(id:string)=>$<HTMLDialogElement>(id);
const clone=<T>(x:T):T=>structuredClone(x);
const clamp=(v:number,lo:number,hi:number)=>Math.max(lo,Math.min(hi,v));
const near=(a:number,b:number)=>Math.abs(a-b)<1e-8;
const green='#397d5c', blue='#587eaa', coral='#d98d73', ink='#344b3d';
let presets:Preset[]=[], scene:Scene, goal:Goal, editorScene:Scene;
let original:Drop|null=null, current:Drop|null=null, prediction:Plan|null=null;
let move=0, frame=0, playing=false, lastTime=0, calls=0, busy=false, ready=false, hasModel=false, revision=0;
let fieldIndex=0, outcomeShown=false, fromHint=false, drag:{id:number;x:number;move:number}|null=null;
let camera={x:0,y:0,scale:1,w:1,h:1,pixel:2};
const names=['The first bounce','A close call','Another way','The quiet corner','Over the hill','The long way','A little detour','One more possibility'];
async function api<T>(path:string,body?:unknown):Promise<T>{
  const response=await fetch(`/api/${path}`,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!response.ok){let message='The local playground could not answer. Please try again.';try{const result=await response.json();message=typeof result.detail==='string'?result.detail:'Check the field settings or JSON file; some values are outside the supported range.';}catch{}throw new Error(message);}
  return response.json() as Promise<T>;
}
function showError(e:unknown){const box=dialog('editorDialog').open?$('editorError'):$('error');let text=e instanceof Error?e.message:String(e);if(text.includes('overlap'))text='That position overlaps the ball or a ramp. Move the bumper back a little, or change the field.';box.textContent=text;box.hidden=false;}
function status(label:string,title:string,text:string,tone=''){
  const box=$('result');box.className=`result ${tone}`;box.replaceChildren();
  for(const[tag,value,cls]of[['span',label,'eyebrow'],['h2',title,''],['p',text,'']]){const el=document.createElement(tag);el.textContent=value;el.className=cls;box.append(el);}
}
function moveText(u:number){return near(u,0)?'0 cm':`${Math.abs(u*100).toFixed(1)} cm ${u<0?'left':'right'}`;}
function sync(){
  $('drop').toggleAttribute('disabled',busy||!ready);
  $('hint').toggleAttribute('disabled',busy||!ready||!hasModel);
  $('hint').title=hasModel?'Let the model imagine a small bumper move.':'No model is loaded. You can still move the bumper and drop the ball.';
  $('hint').textContent=busy?'One moment…':'✧ Give me a hint';
  $('drop').textContent=busy?'One moment…':playing?'Ⅱ Pause bounce':current?(frame>=120?'↻ Drop again':'▶ Continue bounce'):'▶ Drop ball';
  for(const id of ['move','nudgeLeft','nudgeRight','tryTolerance'])$(id).toggleAttribute('disabled',busy||!ready);
  $('resetMove').toggleAttribute('disabled',busy||near(move,0));
  $('moveValue').textContent=moveText(move);$('editValue').textContent=moveText(move);
  if(scene){input('move').value=String(move/scene.ball.radius*100);input('move').setAttribute('aria-valuetext',moveText(move));$('moveLimit').textContent=`Up to ${(scene.ball.radius*100).toFixed(0)} cm each way`;}
  $('marginValue').textContent=prediction?.predicted_margin_m==null?'—':`${(prediction.predicted_margin_m*100).toFixed(1)} cm`;
  $('latencyValue').textContent=prediction?`${(prediction.seconds*1000).toFixed(0)} ms`:'—';
  $('callValue').textContent=String(calls);$('playback').hidden=!current;
  $('play').toggleAttribute('disabled',busy||!current);input('scrub').disabled=busy||!current;
  $('play').textContent=playing?'Ⅱ Pause':frame>=120?'↻ Replay':'▶ Play';
  $('play').setAttribute('aria-label',playing?'Pause this drop':frame>=120?'Replay this drop':'Play this drop');
  input('scrub').value=String(Math.round(frame));$('time').textContent=`${(frame/60).toFixed(2)} / 2.00 s`;
  $('replayKind').textContent=fromHint?'Hint · actual bounce':near(move,0)?'Starting bounce':'Your move · actual bounce';
  $('compareLabel').hidden=!original||near(move,0);$('ghostLabel').hidden=!prediction?.predicted_edited;
  $('bumperControl').hidden=!ready||playing||busy;
  layout();
}
function setMove(value:number){
  if(busy||!ready)return;
  const bounded=clamp(value,-scene.ball.radius,scene.ball.radius);if(near(bounded,move))return;
  revision++;move=near(bounded,0)?0:bounded;current=null;prediction=null;fromHint=false;outcomeShown=false;playing=false;frame=0;lastTime=0;
  $('error').hidden=true;status('YOUR MOVE',near(move,0)?'Back to the beginning.':'A little nudge. A different bounce.',near(move,0)?'Drop the ball to watch the starting bounce.':`Bumper: ${moveText(move)}. Drop the ball to see what changes.`);
  sync();draw();drawLandscape();
}
function loadPreset(p:Preset,index=-1,initialMove=0){
  revision++;scene=clone(p.scene);goal=clone(p.goal);fieldIndex=index;move=clamp(initialMove,-scene.ball.radius,scene.ball.radius);
  original=current=null;prediction=null;frame=0;playing=false;lastTime=0;calls=0;outcomeShown=false;fromHint=false;
  $('error').hidden=true;input('compare').checked=false;input('predictions').checked=true;
  $('fieldName').textContent=index<0?'YOUR OWN LITTLE WORLD':`FIELD ${String(index+1).padStart(2,'0')} · ${names[index%names.length].toUpperCase()}`;
  $('objective').textContent=goal.kind==='avoid'?'Stay out of the red clearing.':'Touch the golden clearing.';
  const interval=goal.start===0&&goal.end===2?'Two seconds to change its path.':`The goal counts from ${goal.start} to ${goal.end} seconds.`;
  $('goalDetail').textContent=`One ball. One little move. ${interval}`;
  status('YOUR TURN','Small move. New possibility.','Slide the coral bumper, then drop the ball. Need an idea? Ask for a hint.');
  renderFields();sync();draw();drawLandscape();
}
async function operation(action:(rev:number)=>Promise<void>){
  if(busy||!ready)return;busy=true;playing=false;lastTime=0;$('error').hidden=true;sync();draw();
  const rev=revision;try{await action(rev);}catch(e){if(rev===revision){status('LET’S TRY AGAIN','That move needs another look.','Check the message below, adjust the move, and try again.');showError(e);}}finally{busy=false;sync();draw();drawLandscape();}
}
async function ensureOriginal(rev:number){
  if(original)return original;
  const data=await api<Drop>('simulate',{scene,goal});if(rev!==revision)return null;
  original=data;calls+=data.simulator_calls??0;return data;
}
function showOutcome(){
  if(!current||outcomeShown)return;outcomeShown=true;
  const success=current.goal.success;
  if(success)status('ACTUAL OUTCOME',near(move,0)?'It works just as it is.':'That little move worked.',near(move,0)?'The starting bounce already meets the goal. No nudge needed.':fromHint?'The hint survived the real bounce. Replay it, compare the paths, or explore another field.':'Your nudge changed the story. Replay it, compare the paths, or explore another field.','good');
  else status('ACTUAL OUTCOME',fromHint?'The hint missed this time.':goal.kind==='avoid'?'A little too close.':'Not quite there.',fromHint?'The dashed path was only a prediction. Try your own nudge and let the next bounce decide.':goal.kind==='avoid'?'The ball entered the red clearing. Try moving the bumper a little.':'The ball missed the golden clearing. Try a different nudge, or ask for a hint.','bad');
}
function startReplay(){
  if(!current||busy)return;if(frame>=120)frame=0;playing=true;lastTime=0;outcomeShown=false;
  status('WATCH THE BOUNCE',near(move,0)?'Here’s where it goes.':'Let’s see what your nudge does.','This is the actual physics. Pause or scrub to take a closer look.');sync();draw();
}
$('drop').addEventListener('click',()=>{
  if(current){if(playing){playing=false;sync();}else startReplay();return;}
  void operation(async rev=>{
    status('DROPPING THE BALL','Here we go.','Checking the bounce in the simulator…');
    let data:Drop|null;
    if(near(move,0)){data=await ensureOriginal(rev);}
    else {
      if(!await ensureOriginal(rev)||rev!==revision)return;
      fromHint=!!prediction&&prediction.edit_m!==null&&near(move,prediction.edit_m);
      data=fromHint?await api<Drop>('verify',{plan_id:prediction!.plan_id}):await api<Drop>('try',{scene,goal,edit_m:move});
      if(rev!==revision)return;calls+=data.new_simulator_calls??data.simulator_calls??0;
    }
    if(rev!==revision||!data)return;current=data;frame=0;outcomeShown=false;playing=true;lastTime=0;
    status('WATCH THE BOUNCE','Here we go.','This is the actual physics. Pause or scrub to take a closer look.');
  });
});
$('hint').addEventListener('click',()=>void operation(async rev=>{
  status('THINKING ABOUT THE BOUNCE','Looking for a little nudge…','The model is imagining possible paths. It might be wrong.');
  if(!await ensureOriginal(rev)||rev!==revision)return;
  const result=await api<Plan>('plan',{scene,goal});if(rev!==revision)return;
  prediction=result;current=null;frame=0;outcomeShown=false;fromHint=false;input('predictions').checked=true;
  if(result.edit_m===null)status('NO HINT FOUND','Your move, explorer.','The model couldn’t find a promising nudge. That doesn’t mean there isn’t one. Try a move yourself.');
  else {move=result.edit_m;
    if(result.status==='known baseline success')status('NO NUDGE NEEDED','Try it just as it is.','The starting bounce already meets the goal. Drop the ball to watch.','good');
    else status('AN IDEA, NOT A PROMISE',`Try ${moveText(move)}.`,`The bumper is in place. The dashed path is imagined; drop the ball to test it.`);
  }
}));
$('play').addEventListener('click',()=>{if(!current||busy)return;if(playing){playing=false;sync();}else startReplay();});
$('scrub').addEventListener('input',()=>{if(!current)return;playing=false;frame=Number(input('scrub').value);if(frame>=120)showOutcome();sync();draw();});
$('move').addEventListener('input',()=>setMove(Number(input('move').value)*scene.ball.radius/100));
$('nudgeLeft').addEventListener('click',()=>setMove(move-.01));$('nudgeRight').addEventListener('click',()=>setMove(move+.01));$('resetMove').addEventListener('click',()=>setMove(0));
for(const id of ['compare','predictions'])$(id).addEventListener('change',draw);
function open(id:string){playing=false;sync();dialog(id).showModal();if(id==='notebookDialog')drawLandscape();}
$('chooseScene').addEventListener('click',()=>open('fieldsDialog'));
$('openHelp').addEventListener('click',()=>open('helpDialog'));$('openNotebook').addEventListener('click',()=>open('notebookDialog'));
document.querySelectorAll<HTMLElement>('[data-close]').forEach(b=>b.addEventListener('click',()=>dialog(b.dataset.close!).close()));
document.querySelectorAll('dialog').forEach(d=>d.addEventListener('click',e=>{if(e.target===d){const r=d.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)d.close();}}));
$('notebookDialog').querySelectorAll('details').forEach(d=>d.addEventListener('toggle',drawLandscape));
function renderFields(){
  const root=$('fields');root.replaceChildren();presets.forEach((p,i)=>{
    const b=document.createElement('button');b.className='field-card';b.setAttribute('aria-current',String(i===fieldIndex));
    const thumb=document.createElement('canvas');thumb.width=84;thumb.height=56;thumb.setAttribute('aria-hidden','true');const ctx=thumb.getContext('2d')!;ctx.fillStyle='#d9e6bd';ctx.fillRect(0,0,84,56);
    const xy=(v:V):V=>[v[0]*14,56-v[1]*14];ctx.strokeStyle='#7e8264';ctx.lineWidth=2;for(const s of p.scene.segments){ctx.beginPath();ctx.moveTo(...xy(s.a));ctx.lineTo(...xy(s.b));ctx.stroke();}
    for(const[v,r,c]of[[p.scene.bumper.position,p.scene.bumper.radius,coral],[p.scene.ball.position,p.scene.ball.radius,'#fffbe9']] as [V,number,string][]){ctx.fillStyle=c;ctx.beginPath();ctx.arc(...xy(v),r*14,0,Math.PI*2);ctx.fill();}
    if(p.goal.region.kind==='disk'){ctx.strokeStyle=p.goal.kind==='avoid'?'#b46c58':'#b39540';ctx.beginPath();ctx.arc(...xy(p.goal.region.center),p.goal.region.radius*14,0,Math.PI*2);ctx.stroke();}
    const text=document.createElement('span'),title=document.createElement('b'),detail=document.createElement('small');title.textContent=`${String(i+1).padStart(2,'0')} · ${names[i%names.length]}`;detail.textContent=p.goal.kind==='reach'?'Touch the clearing':'Stay out of the clearing';text.append(title,detail);b.append(thumb,text);
    b.addEventListener('click',()=>{loadPreset(p,i);dialog('fieldsDialog').close();});root.append(b);
  });
}
function populateEditor(){
  editorScene=clone(scene);$('editorError').hidden=true;
  const values:Record<string,number|string>={ballX:scene.ball.position[0],ballY:scene.ball.position[1],vx:scene.ball.velocity[0],vy:scene.ball.velocity[1],bumperX:scene.bumper.position[0],bumperY:scene.bumper.position[1],goalKind:goal.kind,regionKind:goal.region.kind,goalX:goal.region.center[0],goalY:goal.region.center[1],goalRadius:goal.region.radius,normal:goal.region.normal.join(','),boundary:goal.region.offset/(goal.region.normal.find(v=>v!==0)||1)};
  // Imported half-planes may use any unit normal; retain it without forcing an axis.
  const normal=$<HTMLSelectElement>('normal');if(![...normal.options].some(o=>o.value===values.normal)){const o=document.createElement('option');o.value=String(values.normal);o.textContent='Imported direction';normal.append(o);}
  for(const[k,v]of Object.entries(values))input(k).value=String(v);regionFields();renderSegments();
}
function editField(){dialog('fieldsDialog').close();dialog('notebookDialog').close();populateEditor();open('editorDialog');}
$('openEditor').addEventListener('click',editField);$('editField').addEventListener('click',editField);
function regionFields(){$('diskFields').hidden=input('regionKind').value!=='disk';$('halfFields').hidden=input('regionKind').value!=='halfplane';}
$('regionKind').addEventListener('change',regionFields);
function renderSegments(){
  const root=$('segments');root.replaceChildren();$('segmentCount').textContent=`(${editorScene.segments.length}/4)`;$('addSegment').toggleAttribute('disabled',editorScene.segments.length>=4);
  editorScene.segments.forEach((s,i)=>{const div=document.createElement('div');div.className='segment';const title=document.createElement('small');title.textContent=`Ramp ${i+1}`;div.append(title);
    for(const point of ['a','b'] as const){const row=document.createElement('div');row.className='pair';for(let j=0;j<2;j++){const label=document.createElement('label');label.textContent=`${point.toUpperCase()} ${j===0?'x':'y'}`;const f=document.createElement('input');f.type='number';f.step='any';f.min='0';f.max=j===0?'6':'4';f.required=true;f.value=String(s[point][j]);f.addEventListener('input',()=>s[point][j]=f.value===''?NaN:Number(f.value));label.append(f);row.append(label);}div.append(row);}
    const remove=document.createElement('button');remove.type='button';remove.className='text-button';remove.textContent='Remove ramp';remove.addEventListener('click',()=>{editorScene.segments.splice(i,1);renderSegments();});div.append(remove);root.append(div);
  });
}
$('addSegment').addEventListener('click',()=>{if(editorScene.segments.length>=4)return;editorScene.segments.push({id:`segment-${Date.now()}`,a:[3.5,1.2],b:[4.5,.8],radius:.04,material:{friction:.4,restitution:.7}});renderSegments();});
function readEditor(){
  const n=(id:string)=>{const v=Number(input(id).value);if(input(id).value===''||!Number.isFinite(v))throw new Error('Please enter a number in every field.');return v;};
  const s=clone(editorScene),g=clone(goal);s.family='authored';s.ball.position=[n('ballX'),n('ballY')];s.ball.velocity=[n('vx'),n('vy')];s.bumper.position=[n('bumperX'),n('bumperY')];
  g.kind=input('goalKind').value;g.region.kind=input('regionKind').value;g.region.center=[n('goalX'),n('goalY')];g.region.radius=n('goalRadius');g.region.normal=input('normal').value.split(',').map(Number) as V;g.region.offset=n('boundary')*(g.region.normal.find(v=>v!==0)||1);
  return {scene:s,goal:g};
}
$('editorForm').addEventListener('submit',async e=>{e.preventDefault();if(busy)return;const rev=revision;$('applyField').setAttribute('disabled','');try{const task=readEditor();await api('validate',task);if(rev!==revision)return;loadPreset({name:'Your field',...task});dialog('editorDialog').close();}catch(e){showError(e);}finally{$('applyField').removeAttribute('disabled');}});
$('save').addEventListener('click',async()=>{try{const task=readEditor();await api('validate',task);const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(task,null,2)],{type:'application/json'}));a.download='counterfactual-field.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}catch(e){showError(e);}});
$('loadButton').addEventListener('click',()=>input('load').click());
$('load').addEventListener('change',async()=>{const rev=revision;try{const file=input('load').files?.[0];if(!file)return;const parsed=JSON.parse(await file.text()),task={scene:parsed.scene,goal:parsed.goal};await api('validate',task);if(rev!==revision)return;loadPreset({name:'Your field',...task});dialog('editorDialog').close();}catch(e){showError(e);}finally{input('load').value='';}});

// Only rasterization uses integer pixels. Every physical coordinate stays in SI units.
function layout(){
  const canvas=$<HTMLCanvasElement>('scene'),r=$('world').getBoundingClientRect();
  const w=Math.ceil(r.width/2),h=Math.ceil(r.height/2);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
  const pixel=r.width/w,small=r.width<=700,top=(small?150:175)/pixel,bottom=(small?280:245)/pixel;
  const scale=Math.max(8,Math.min((w-38)/6,(h-top-bottom)/4));
  camera={w,h,pixel,scale,x:(w-scale*6)/2,y:top+(h-top-bottom-scale*4)/2+4*scale};
  const box=document.querySelector('.dialogue')!.getBoundingClientRect();
  document.querySelector<HTMLElement>('.field-tools')!.style.bottom=`${r.bottom-box.top+14}px`;
  if(scene){
    const control=$('bumperControl'),cw=small?190:210,ch=control.offsetHeight||76;
    const [px,py]=xy([scene.bumper.position[0]+move,scene.bumper.position[1]]),x=px*pixel,y=py*pixel,br=scene.bumper.radius*scale*pixel;
    const toRect=(p:number[],radius:number)=>{const[a,b]=xy(p),rr=radius*scale*pixel+8;return {x:a*pixel-rr,y:b*pixel-rr,w:rr*2,h:rr*2};};
    const ball=toRect(scene.ball.position,scene.ball.radius),target=goal.region.kind==='disk'?toRect(goal.region.center,goal.region.radius):null;
    const proposals=[{x:x+br+18,y:y-ch/2,side:'right'},{x:x-br-18-cw,y:y-ch/2,side:'left'},
      {x:x-cw/2,y:y+br+18,side:'below'},{x:x-cw/2,y:y-br-18-ch,side:'above'},
      {x:x-cw/2,y:target?target.y+target.h+7:y+br+18,side:'below'}];
    const maxY=Math.max(155,box.top-r.top-52-ch);
    const overlap=(a:{x:number;y:number},b:{x:number;y:number;w:number;h:number})=>Math.max(0,Math.min(a.x+cw,b.x+b.w)-Math.max(a.x,b.x))*Math.max(0,Math.min(a.y+ch,b.y+b.h)-Math.max(a.y,b.y));
    const bumper=toRect([scene.bumper.position[0]+move,scene.bumper.position[1]],scene.bumper.radius);
    const choices=proposals.map(p=>({...p,x:clamp(p.x,12,r.width-cw-12),y:clamp(p.y,155,maxY)}));
    const score=(p:typeof choices[number])=>overlap(p,ball)*20+(target?overlap(p,target)*20:0)+overlap(p,bumper)*10+Math.hypot(p.x+cw/2-x,p.y+ch/2-y);
    choices.sort((a,b)=>score(a)-score(b));const chosen=choices[0];
    control.style.left=`${chosen.x}px`;control.style.top=`${chosen.y}px`;control.dataset.side=chosen.side;
  }
}
function xy(p:number[]):V{return[camera.x+p[0]*camera.scale,camera.y-p[1]*camera.scale];}
function disk(ctx:CanvasRenderingContext2D,x:number,y:number,r:number,fill:string,outline?:string){
  x=Math.round(x);y=Math.round(y);r=Math.max(1,Math.round(r));if(!Number.isFinite(x+y+r)||x+r<0||x-r>camera.w||y+r<0||y-r>camera.h)return;
  ctx.fillStyle=outline??fill;for(let row=-r;row<=r;row++){const q=Math.floor(Math.sqrt(Math.max(0,r*r-row*row)));ctx.fillRect(x-q,y+row,2*q+1,1);}
  if(outline&&r>1){ctx.fillStyle=fill;for(let row=1-r;row<r;row++){const q=Math.floor(Math.sqrt(Math.max(0,(r-1)*(r-1)-row*row)));ctx.fillRect(x-q,y+row,2*q+1,1);}}
}
function line(ctx:CanvasRenderingContext2D,a:V,b:V,color:string,width=1,dashed=false,phase=0){
  if(![...a,...b].every(Number.isFinite))return phase;
  const count=Math.max(1,Math.ceil(Math.max(Math.abs(b[0]-a[0]),Math.abs(b[1]-a[1]))));
  // Skip entirely offscreen lines and bound pathological imported/out-of-frame paths.
  if(count>20000||(a[0]<0&&b[0]<0)||(a[0]>camera.w&&b[0]>camera.w)||(a[1]<0&&b[1]<0)||(a[1]>camera.h&&b[1]>camera.h))return phase+count;
  ctx.fillStyle=color;for(let j=0;j<=count;j++)if(!dashed||((phase+j)%9)<4)ctx.fillRect(Math.round(a[0]+(b[0]-a[0])*j/count)-Math.floor(width/2),Math.round(a[1]+(b[1]-a[1])*j/count)-Math.floor(width/2),width,width);
  return phase+count;
}
function meadow(ctx:CanvasRenderingContext2D){
  const{w,h}=camera;ctx.fillStyle='#dce8bf';ctx.fillRect(0,0,w,h);
  // Fixed little patches and tufts: quiet, deterministic scenery, never collision geometry.
  for(let y=9;y<h;y+=22)for(let x=8;x<w;x+=27){const seed=(x*17+y*31)%97,px=x+seed%13,py=y+seed%7;ctx.fillStyle=seed%3===0?'#cdddac':'#d3e1b4';ctx.fillRect(px-5,py,11,3);ctx.fillStyle='#afc59088';ctx.fillRect(px,py-2,1,3);ctx.fillRect(px+2,py-1,1,2);}
  // Border gardens keep the central field clear and make the world fill the window.
  for(let i=0;i<Math.ceil(w/23);i++){for(const edge of [0,1]){const x=i*23+(i*7)%11,y=edge?h-8-(i*11)%16:7+(i*13)%18;ctx.fillStyle='#94b17a';ctx.fillRect(x,y,2,5);ctx.fillRect(x-2,y+3,6,1);ctx.fillStyle=i%3===0?'#f4dab0':'#f2f1cd';ctx.fillRect(x-1,y-2,4,3);ctx.fillStyle='#bd9e64';ctx.fillRect(x,y-1,1,1);}}
  for(let i=0;i<10;i++){const x=i%2?camera.w-11:7,y=55+i*19;ctx.fillStyle='#a4be89';ctx.fillRect(x,y,4,6);ctx.fillRect(x-2,y+2,8,3);ctx.fillStyle='#bdd19e';ctx.fillRect(x,y+1,2,2);}
  const left=camera.x,top=camera.y-4*camera.scale;
  ctx.strokeStyle='#a9bc8a66';ctx.setLineDash([2,6]);ctx.strokeRect(Math.round(left-5),Math.round(top-5),Math.round(6*camera.scale+10),Math.round(4*camera.scale+10));ctx.setLineDash([]);
}
function goalArt(ctx:CanvasRenderingContext2D){
  const avoid=goal.kind==='avoid',border=avoid?'#b36b5e':'#b29b50',fill=avoid?'#dfb29c99':'#edda8b99';
  if(goal.region.kind==='disk'){
    const[x,y]=xy(goal.region.center),r=goal.region.radius*camera.scale;disk(ctx,x,y,r,fill);
    for(let i=0;i<16;i++){const a=i*Math.PI/8;ctx.fillStyle=border;ctx.fillRect(Math.round(x+Math.cos(a)*r)-1,Math.round(y+Math.sin(a)*r)-1,2,2);}
    ctx.fillStyle=border;ctx.fillRect(Math.round(x)-1,Math.round(y)-6,1,11);ctx.fillStyle=avoid?'#c17a66':'#d0a550';ctx.fillRect(Math.round(x),Math.round(y)-6,6,4);
    if(avoid){ctx.fillStyle='#fff2d7';ctx.fillRect(Math.round(x)+2,Math.round(y)-5,2,1);}
  }else{
    const n=goal.region.normal,o=goal.region.offset;let poly:V[]=[[0,0],[6,0],[6,4],[0,4]],result:V[]=[];
    for(let i=0;i<poly.length;i++){const a=poly[i],b=poly[(i+1)%poly.length],da=n[0]*a[0]+n[1]*a[1]-o,db=n[0]*b[0]+n[1]*b[1]-o;if(da>=0)result.push(a);if((da>=0)!==(db>=0)){const t=da/(da-db);result.push([a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])]);}}
    if(result.length){ctx.beginPath();result.forEach((p,i)=>{const[x,y]=xy(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.closePath();ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=border;ctx.setLineDash([3,3]);ctx.stroke();ctx.setLineDash([]);}
  }
}
function draw(){
  if(!scene)return;const canvas=$<HTMLCanvasElement>('scene'),ctx=canvas.getContext('2d')!;ctx.imageSmoothingEnabled=false;meadow(ctx);goalArt(ctx);
  for(const s of scene.segments){const a=xy(s.a),b=xy(s.b),r=Math.max(2,Math.round(s.radius*camera.scale));
    line(ctx,[a[0]+1,a[1]+3],[b[0]+1,b[1]+3],'#81936755',r*2+2);line(ctx,a,b,'#656b56',r*2+1);disk(ctx,...a,r,'#656b56');disk(ctx,...b,r,'#656b56');line(ctx,[a[0],a[1]-1],[b[0],b[1]-1],'#a3aa7f',Math.max(1,r*2-2));
    const dist=Math.hypot(b[0]-a[0],b[1]-a[1]);for(let d=6;d<dist;d+=13){const t=d/dist;ctx.fillStyle='#697b51';ctx.fillRect(Math.round(a[0]+(b[0]-a[0])*t)-1,Math.round(a[1]+(b[1]-a[1])*t)-r-1,3,2);}
  }
  const point=[scene.bumper.position[0]+move,scene.bumper.position[1]], [bx,by]=xy(point),br=scene.bumper.radius*camera.scale;
  const base=xy(scene.bumper.position),left=xy([scene.bumper.position[0]-scene.ball.radius,scene.bumper.position[1]]),right=xy([scene.bumper.position[0]+scene.ball.radius,scene.bumper.position[1]]);
  if(!playing){line(ctx,[left[0]-br-5,by],[right[0]+br+5,by],'#a98769',1,true);for(const x of [left[0]-br-5,right[0]+br+5])line(ctx,[x,by-3],[x,by+3],'#a98769');}
  if(!near(move,0)){ctx.globalAlpha=.35;disk(ctx,...base,br,'#f2e4c0','#9c8e71');ctx.globalAlpha=1;}
  disk(ctx,bx+1,by+3,br,'#71825b44');disk(ctx,bx,by,br,coral,'#825f54');disk(ctx,bx-2,by-2,Math.max(1,br-4),'#edb08c');ctx.fillStyle='#fae1b8';ctx.fillRect(Math.round(bx-br*.45),Math.round(by-br*.5),Math.max(2,Math.round(br*.55)),2);ctx.fillStyle='#c17e68';ctx.fillRect(Math.round(bx)-1,Math.round(by)-2,2,5);
  const path=(states:number[][],color:string,dashed=false,upto=states.length-1)=>{let phase=0;for(let j=1;j<=Math.min(upto,states.length-1);j++)phase=line(ctx,xy(states[j-1]),xy(states[j]),color,1,dashed,phase);};
  const index=Math.min(120,Math.floor(frame));
  if(input('compare').checked&&original&&!near(move,0)){path(original.trajectory.states,blue+'a0',false,current?index:120);}
  if(prediction?.predicted_edited&&input('predictions').checked)path(prediction.predicted_edited,green+'c0',true);
  if(current)path(current.trajectory.states,green,false,index);
  const ball=(p:number[],ghost=false)=>{const[x,y]=xy(p),r=scene.ball.radius*camera.scale;ctx.globalAlpha=ghost?.5:1;disk(ctx,x+1,y+2,r,'#81936755');disk(ctx,x,y,r,ghost?'#b9d3df':'#fff7d9',ghost?blue:ink);ctx.fillStyle=ghost?blue:ink;if(r>=4){ctx.fillRect(Math.round(x)-2,Math.round(y)-1,1,2);ctx.fillRect(Math.round(x)+2,Math.round(y)-1,1,2);}ctx.globalAlpha=1;};
  if(input('compare').checked&&original&&!near(move,0))ball(original.trajectory.states[index],true);
  ball(current?current.trajectory.states[Math.min(index,current.trajectory.states.length-1)]:scene.ball.position);
  if(!current){const[x,y]=xy(scene.ball.position);ctx.font='bold 5px monospace';ctx.textAlign='center';ctx.fillStyle='#6b7c58';ctx.fillText('THE BALL',Math.round(x),Math.round(y-scene.ball.radius*camera.scale-7));ctx.textAlign='left';}
  if(current&&frame>=120){const last=current.trajectory.states.at(-1)!;if(last[0]<0||last[0]>6||last[1]<0||last[1]>4){const[x,y]=xy([clamp(last[0],.12,5.88),clamp(last[1],.12,3.88)]);ctx.fillStyle='#6b7c58';ctx.font='6px monospace';ctx.fillText('ball left the field',clamp(x-35,3,camera.w-100),y-3);}}
}
const sceneCanvas=$<HTMLCanvasElement>('scene');
sceneCanvas.addEventListener('pointerdown',e=>{
  if(busy||!ready||playing)return;const r=sceneCanvas.getBoundingClientRect(),[x,y]=xy([scene.bumper.position[0]+move,scene.bumper.position[1]]);
  if(Math.hypot((e.clientX-r.left)/camera.pixel-x,(e.clientY-r.top)/camera.pixel-y)>Math.max(scene.bumper.radius*camera.scale+5,15))return;
  drag={id:e.pointerId,x:e.clientX,move};sceneCanvas.setPointerCapture(e.pointerId);sceneCanvas.style.cursor='grabbing';e.preventDefault();
});
sceneCanvas.addEventListener('pointermove',e=>{
  if(drag&&drag.id===e.pointerId){setMove(drag.move+(e.clientX-drag.x)/(camera.scale*camera.pixel));return;}
  if(!scene)return;const r=sceneCanvas.getBoundingClientRect(),[x,y]=xy([scene.bumper.position[0]+move,scene.bumper.position[1]]);sceneCanvas.style.cursor=!playing&&!busy&&Math.hypot((e.clientX-r.left)/camera.pixel-x,(e.clientY-r.top)/camera.pixel-y)<Math.max(scene.bumper.radius*camera.scale+5,15)?'grab':'default';
});
for(const name of ['pointerup','pointercancel','lostpointercapture'])sceneCanvas.addEventListener(name,()=>{drag=null;sceneCanvas.style.cursor='default';});
function drawLandscape(){
  if(!dialog('notebookDialog').open)return;const canvas=$<HTMLCanvasElement>('landscape'),r=canvas.getBoundingClientRect();if(!r.width||!r.height)return;
  const dpr=window.devicePixelRatio||1;canvas.width=r.width*dpr;canvas.height=r.height*dpr;const ctx=canvas.getContext('2d')!;ctx.scale(dpr,dpr);const w=r.width,h=r.height,left=42,right=w-12,top=12,bottom=h-25;
  ctx.font='10px monospace';ctx.fillStyle='#65755b';ctx.strokeStyle='#a3b48d';ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,bottom);ctx.lineTo(right,bottom);ctx.stroke();for(const x of [-1,0,1])ctx.fillText(String(x),left+(x+1)/2*(right-left)-3,bottom+16);
  if(!prediction){ctx.fillText('Ask for a hint to see the predicted possibilities.',left+8,(top+bottom)/2);return;}
  const values=prediction.margins_m.filter((v):v is number=>v!==null);if(!values.length)return;
  const min=Math.min(-.04,...values),max=Math.max(.06,...values),y=(v:number)=>bottom-(v-min)/(max-min)*(bottom-top),x=(v:number)=>left+(v/scene.ball.radius+1)/2*(right-left);
  for(const v of [min,0,max]){ctx.fillStyle='#6b765f';ctx.fillText(v.toFixed(2),0,y(v)+3);ctx.strokeStyle='#e1e5d4';ctx.beginPath();ctx.moveTo(left,y(v));ctx.lineTo(right,y(v));ctx.stroke();}
  ctx.strokeStyle=blue;ctx.lineWidth=1.7;ctx.beginPath();let connected=false;prediction.grid_m.forEach((u,i)=>{const v=prediction!.margins_m[i];if(v===null){connected=false;return;}if(!connected)ctx.moveTo(x(u),y(v));else ctx.lineTo(x(u),y(v));connected=true;});ctx.stroke();
  ctx.setLineDash([3,3]);ctx.strokeStyle='#8d987c';ctx.beginPath();ctx.moveTo(left,y(.03));ctx.lineTo(right,y(.03));ctx.stroke();ctx.setLineDash([]);if(prediction.edit_m!==null){ctx.strokeStyle='#be785e';ctx.beginPath();ctx.moveTo(x(prediction.edit_m),top);ctx.lineTo(x(prediction.edit_m),bottom);ctx.stroke();}
}
async function evidence(info:{model:{name:string}|null;report_available:boolean}){
  $('modelName').textContent=info.model?`Local model: ${info.model.name}. Hints predict; actual bounces use the simulator.`:'No model is loaded. You can still move the bumper and play; hints need a checkpoint.';
  if(!info.report_available)return;
  try{const report=await api<{initially_failing_grid_solvable:number;test_parents:number;metrics:Record<string,{estimate:number|null;ci95:number[]}>}>('report');const success=report.metrics.verified_success_solvable.estimate;
    $('evidenceText').textContent=`This loaded checkpoint repaired ${success===null?'an undefined fraction':(success*100).toFixed(1)+'%'} of ${report.initially_failing_grid_solvable} grid-solvable development tasks across ${report.test_parents} parents. These repeatedly inspected tasks are not a fresh final test. The figures below identify their own cohorts and training seeds.`;
    const names=info.model?.name==='Contact mechanics experiment'?[['mechanics-comparison','Mechanics and training, three seeds. Validation locality and contact timing beside development repairs.'],['mechanics-budgets','Absolute failures, proposal coverage, and training exposure.'],['selection-calibration','Selection calibration and an independent audit; uncertainty remains.']]:info.model?.name.startsWith('A recurrent')?[['representation-comparison','Initial recurrent A: decision gains alongside locality failures.'],['pilot-summary','Development outcomes and failed verifications.']]:[['pilot-summary','Baseline intervention outcomes.'],['numerical-audit','Simulation step-halving audit.']];
    for(const[name,caption]of names){const fig=document.createElement('figure'),img=document.createElement('img'),cap=document.createElement('figcaption'),a=document.createElement('a');img.src=`/figures/${name}.png`;img.alt=caption;img.loading='lazy';img.onerror=()=>fig.remove();cap.textContent=caption;a.href=`/figures/${name}.svg`;a.textContent='Vector SVG ↗';cap.append(a);fig.append(img,cap);$('figures').append(fig);}
  }catch{$('evidenceText').textContent='The research record could not be loaded. The playground still works.';}
}
async function architectureEvidence(){
  try{
    const record=await api<{summary:{protocol:{parent_pools:{test:[number,number]}}};figures:string[]}>('research');
    $('architectureResults').hidden=false;
    $('architectureText').textContent=`Four architectures, three training seeds, ${record.summary.protocol.parent_pools.test[1]} fresh final parents, and an analytic-response control. Circles show seed results; uncertainty resamples whole parents. The hint model in this playground is unchanged. Read successes alongside failures, compute cost, and embedding rank.`;
    const captions:Record<string,string>={
      'architecture-comparison':'Fresh-cohort repairs, intervention-effect error, and ramp event readout. Three seeds per learned variant.',
      'architecture-effects':'Paired effects separate explicit difference prediction from joint-embedding training. Runtime includes cold candidate forecasts.',
      'architecture-calibration':'Fixed versus calibrated acceptance on fresh final parents. Calibration is empirical, not a risk guarantee.',
      'architecture-learning':'Validation learning and embedding diagnostics. Shading spans three seeds, not a confidence interval.',
      'architecture-context':'D seed-0 on two fixed diagnostic fixtures: anticipatory context may change before physical influence. Internal norm is not an edit cost.',
      'architecture-contact-audit':'Post-evaluation diagnosis on exact validation states: the learned phase readout misses contacts at its fixed threshold. Geometric admission remains accurate. No weights or thresholds were changed.',
      'architecture-success':'First successful A seed-0 proposal in the frozen final order. Actual paths are solid; predictions are dashed.',
      'architecture-failure':'First failed A seed-0 proposal in the same frozen order, with the same plotting treatment.',
      'architecture-contact-boundary':'First final scene where edits change bumper contact existence. Unsmooth transitions are retained.'
    };
    for(const name of record.figures){
      const fig=document.createElement('figure'),img=document.createElement('img'),cap=document.createElement('figcaption'),link=document.createElement('a');
      img.src=`/research-figures/${name}.png`;img.alt=captions[name]??name;img.loading='lazy';cap.textContent=img.alt;
      link.href=`/research-figures/${name}.svg`;link.textContent='Vector SVG ↗';cap.append(link);fig.append(img,cap);$('architectureFigures').append(fig);
    }
  }catch{/* A research run is optional; missing records do not interrupt play. */}
}
async function contactEvidence(){
  type Example = {scene:Scene;goal:Goal;edit_m:number;epsilon_m:number;common_component:{low_m:number;high_m:number}|null;explanation:string;qualifier:string};
  try{
    const record=await api<{example:Example|null;figures:string[]}>('contact-research');
    $('contactResults').hidden=false;
    $('contactText').textContent='Contact learning, episode history, integration, and tolerance are tested separately. Checking a neighborhood improves reliability with both learned and analytic predictions; a learned-model advantage is not established. The active hint model stays the same.';
    const example=record.example;
    if(example?.common_component){
      const range=example.common_component,mm=(x:number)=>(x*1000).toFixed(3).replace(/0+$/,'').replace(/\.$/,'');
      $('toleranceExampleText').textContent=`Study example: ${mm(example.edit_m)} mm right. Successful grid positions span ${mm(range.low_m)}–${mm(range.high_m)} mm right at both tested simulator rates. All five offsets within ±${mm(example.epsilon_m)} mm pass. ${example.explanation} ${example.qualifier}`;
      $('tryTolerance').hidden=false;
      $('tryTolerance').addEventListener('click',()=>{
        if(busy||!ready)return;
        loadPreset({name:'A little room to spare',scene:example.scene,goal:example.goal},-1,example.edit_m);
        dialog('notebookDialog').close();$('fieldName').textContent='A LITTLE ROOM TO SPARE';
        status('MEASURED EXAMPLE','A little room for error.',`${mm(example.edit_m)} mm right, with five tested offsets across ±${mm(example.epsilon_m)} mm. Drop the ball to check this move yourself, or explore the edge.`);
      });
    }
    const captions:Record<string,string>={
      'contact-training':'A-only diagnosis, three seeds: raw event learning alongside physical accuracy and gradient scale.',
      'contact-mask-control':'Raw precision and the always-positive geometric-mask control. A good masked score alone does not establish event learning.',
      'contact-episodes':'Actual begin/separate events and active contact. Bands span three seeds; competing contacts remain underrepresented.',
      'contact-integration':'Separate analytic subdivision control: recovered onsets, extra onset chatter, and computation cost.',
      'tolerance-selection':'Fresh final parents, common margin: five-offset success, proposal risk and coverage. Edit costs are paired on tasks proposed by both rules.',
      'tolerance-calibration':'Paired parent bootstrap intervals and calibrated proposal counts. Abstentions count as unsuccessful; empirical calibration is not a risk certificate.',
      'tolerance-success':'Fixed-order successful illustration: actual sampled edit landscape, prediction, and five execution offsets.',
      'tolerance-failure':'Fixed-order failed illustration, shown with the same treatment. A predicted window can fail at its center.',
      'tolerance-events':'Boundary diagnosis at two physics rates: stable sampled goal labels coexist with timestep-sensitive repeated contact. This is association, not unique causal attribution.'
    };
    for(const name of record.figures){
      const fig=document.createElement('figure'),img=document.createElement('img'),cap=document.createElement('figcaption'),link=document.createElement('a');
      img.src=`/contact-figures/${name}.png`;img.alt=captions[name]??name;img.loading='lazy';cap.textContent=img.alt;
      link.href=`/contact-figures/${name}.svg`;link.textContent='Vector SVG ↗';cap.append(link);fig.append(img,cap);$('contactFigures').append(fig);
    }
  }catch{/* Optional saved research never blocks the playground. */}
}
window.addEventListener('resize',()=>{layout();draw();drawLandscape();});
document.addEventListener('visibilitychange',()=>{if(document.hidden){playing=false;lastTime=0;sync();}});
function animate(t:number){if(playing&&current){if(lastTime)frame=Math.min(120,frame+Math.min(t-lastTime,100)*.06);lastTime=t;if(frame>=120){playing=false;showOutcome();}sync();draw();}requestAnimationFrame(animate);}
async function init(){
  try{const info=await api<{model:{name:string}|null;report_available:boolean}>('info');presets=await api<Preset[]>('presets');if(!presets.length)throw new Error('No fields are available. Restart the local playground.');hasModel=!!info.model;ready=true;loadPreset(presets[0],0);void evidence(info);void architectureEvidence();void contactEvidence();requestAnimationFrame(animate);}
  catch(e){showError(e);status('CAN’T REACH THE PLAYGROUND','The field is taking a rest.','Start the local server, then reload this page.');}
}
void init();
