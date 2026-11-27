// Execute the shipped UI against a small event/DOM harness. This verifies flow,
// not browser layout, CSS rendering, native focus, or pointer hit-testing.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const html=fs.readFileSync(new URL('../index.html',import.meta.url),'utf8');
const source=fs.readFileSync(new URL('../dist/app.js',import.meta.url),'utf8').replace(/export\s*\{\s*\};?/g,'');
const scene={schema_version:1,id:'fixture',parent_id:'fixture',family:'approach',seed:0,ball:{id:'ball',position:[1.2,3.3],velocity:[2.4,0],radius:.12},bumper:{id:'bumper',position:[2.5,1.7],radius:.32},segments:[]};
const goal={ball_id:'ball',kind:'reach',region:{kind:'disk',center:[3,.5],radius:.3,normal:[1,0],offset:4},start:0,end:2};
const states=Array.from({length:121},(_,i)=>[1.2+i/100,3.3-i/60,2.4,-i/60,0,0]);
const trajectory={times:states.map((_,i)=>i/60),states,events:[]};
const plan={plan_id:'hint-1',status:'predicted success',edit_m:-.06,predicted_margin_m:.04,predicted_original:states,predicted_edited:states,candidate_count:257,seconds:.1,grid_m:[-.12,0,.12],margins_m:[.04,-.2,.05]};
async function boot({model=true,width=1400,height=900,holdPlan=false,failTry=false,contactExample=false}={}){
  const elements=new Map(),calls=[],raf=[];let releasePlan;
  class Element{
    constructor(tag='div',id=''){this.tagName=tag.toUpperCase();this.id=id;this.value='';this.children=[];this.options=[];this.dataset={};this.attrs={};this.style={};this.events={};this.hidden=false;this.disabled=false;this.checked=false;this.open=false;this.textContent='';}
    addEventListener(type,fn){(this.events[type]??=[]).push(fn);}
    async fire(type,event={}){for(const fn of this.events[type]??[])await fn({target:this,preventDefault(){},...event});}
    click(){if(!this.disabled)return this.fire('click');}
    append(...nodes){this.children.push(...nodes);if(this.tagName==='SELECT')this.options.push(...nodes);}
    replaceChildren(...nodes){this.children=[...nodes];}
    remove(){}
    setAttribute(k,v){this.attrs[k]=v;if(k==='disabled')this.disabled=true;}
    removeAttribute(k){delete this.attrs[k];if(k==='disabled')this.disabled=false;}
    toggleAttribute(k,on){on?this.setAttribute(k,''):this.removeAttribute(k);}
    getAttribute(k){return this.attrs[k];}
    showModal(){this.open=true;}
    close(){this.open=false;}
    setPointerCapture(){}
    getBoundingClientRect(){
      if(this.id==='world'||this.id==='scene')return {left:0,top:0,bottom:height,width,height};
      if(this.id==='dialogue')return {left:10,top:height-(width<=700?260:228),bottom:height-38,width:Math.min(width-28,880),height:width<=700?230:190};
      return {left:0,top:0,bottom:200,width:300,height:200};
    }
    getContext(){return new Proxy({}, {get:(o,k)=>o[k]??(()=>{}),set:(o,k,v)=>(o[k]=v,true)});}
    querySelectorAll(){return [];}
  }
  for(const match of html.matchAll(/<([\w]+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)){
    const e=new Element(match[1],match[3]),a=match[2];e.hidden=/\bhidden\b/.test(a);e.disabled=/\bdisabled\b/.test(a);e.checked=/\bchecked\b/.test(a);e.value=a.match(/\bvalue="([^"]*)"/)?.[1]??'';elements.set(e.id,e);
  }
  for(const id of ['dialogue','field-tools'])elements.set(id,new Element('div',id));
  for(const v of ['1,0','-1,0','0,1','0,-1']){const e=new Element('option');e.value=v;elements.get('normal').append(e);}
  const closers=[...html.matchAll(/data-close="([^"]+)"/g)].map(m=>{const e=new Element('button');e.dataset.close=m[1];return e;});
  const document={hidden:false,getElementById:id=>elements.get(id)??null,createElement:tag=>new Element(tag),querySelector:s=>elements.get(s.slice(1)),querySelectorAll:s=>s==='dialog'?[...elements.values()].filter(e=>e.tagName==='DIALOG'):s==='[data-close]'?closers:[],addEventListener(){}};
  const presets=[{name:'one',scene,goal},{name:'two',scene:{...scene,id:'second'},goal:{...goal,kind:'avoid'}}];
  const fetch=async(path,options={})=>{
    const name=path.replace('/api/',''),body=options.body?JSON.parse(options.body):undefined;calls.push({name,body});
    let data,status=200;
    if(name==='info')data={model:model?{name:'fixture model'}:null,report_available:false};
    else if(name==='presets')data=presets;
    else if(name==='contact-research'&&contactExample)data={figures:[],example:{scene:{...scene,id:'measured'},goal,edit_m:.061875,epsilon_m:.005,common_component:{low_m:.05625,high_m:.12},explanation:'Repeated contact changes.',qualifier:'Sampled, not continuous.'}};
    else if(name==='simulate')data={trajectory,goal:{success:false,margin:-.1},simulator_calls:1};
    else if(name==='try'){
      if(failTry){status=422;data={detail:'initial bumper/ramp overlap'};}
      else data={trajectory,goal:{success:true,margin:.04},simulator_calls:1};
    }else if(name==='plan'){
      if(holdPlan)await new Promise(resolve=>{releasePlan=resolve;});data=plan;
    }else if(name==='verify')data={trajectory,goal:{success:true,margin:.03},new_simulator_calls:1};
    else if(name==='validate')data={valid:true,simulator_calls:0};
    else throw new Error(`Unexpected API ${name}`);
    return {ok:status===200,status,json:async()=>structuredClone(data)};
  };
  const context=vm.createContext({document,window:{devicePixelRatio:1,addEventListener(){}},fetch,structuredClone,requestAnimationFrame:fn=>raf.push(fn),setTimeout,URL:{createObjectURL:()=>'',revokeObjectURL(){}},Blob,console});
  vm.runInContext(source,context);
  const settle=async()=>{for(let i=0;i<8;i++)await new Promise(setImmediate);};
  await settle();
  return {elements,calls,context,settle,release:()=>releasePlan?.(),click:async id=>{await elements.get(id).click();await settle();},tick:async time=>{const next=raf.shift();assert.ok(next);next(time);await settle();},finish:async function(){for(let i=1;i<=22;i++)await this.tick(i*100);},move:async value=>{elements.get('move').value=String(value);await elements.get('move').fire('input');await settle();}};
}
const count=(app,name)=>app.calls.filter(c=>c.name===name).length;
const heading=app=>app.elements.get('result').children[1].textContent;

test('no empty replay; drop autoplays, pauses, scrubs, and replays without new simulation',async()=>{
  const a=await boot();assert.equal(a.elements.get('playback').hidden,true);assert.equal(a.elements.get('play').disabled,true);
  await a.click('play');assert.equal(count(a,'simulate'),0);
  await a.click('drop');assert.equal(count(a,'simulate'),1);assert.equal(a.elements.get('playback').hidden,false);assert.match(a.elements.get('drop').textContent,/Pause/);
  await a.tick(100);await a.tick(200);assert.equal(a.elements.get('time').textContent,'0.10 / 2.00 s');
  await a.click('play');a.elements.get('scrub').value='60';await a.elements.get('scrub').fire('input');assert.equal(a.elements.get('time').textContent,'1.00 / 2.00 s');
  await a.click('play');await a.finish();assert.match(heading(a),/Not quite/);
  await a.click('play');assert.equal(a.elements.get('time').textContent,'0.00 / 2.00 s');assert.equal(count(a,'simulate'),1);
});

test('manual nudge uses bounded manual simulation; moving again invalidates replay',async()=>{
  const a=await boot({width:390,height:844});await a.click('nudgeLeft');assert.equal(a.elements.get('moveValue').textContent,'1.0 cm left');
  await a.click('drop');assert.equal(count(a,'try'),1);assert.equal(a.calls.find(c=>c.name==='try').body.edit_m,-.01);assert.equal(count(a,'simulate'),1);assert.equal(count(a,'verify'),0);
  await a.finish();assert.match(heading(a),/worked/);assert.equal(a.elements.get('compareLabel').hidden,false);
  await a.move(1000);assert.equal(a.elements.get('moveValue').textContent,'12.0 cm right');assert.equal(a.elements.get('playback').hidden,true);assert.equal(a.elements.get('play').disabled,true);
  await a.click('resetMove');assert.equal(a.elements.get('moveValue').textContent,'0 cm');assert.equal(a.elements.get('compareLabel').hidden,true);
});

test('hint is not a verification; dropping tests once; manual change discards the hint',async()=>{
  const a=await boot();await a.click('hint');assert.equal(count(a,'verify'),0);assert.equal(count(a,'try'),0);assert.equal(a.elements.get('moveValue').textContent,'6.0 cm left');assert.equal(a.elements.get('ghostLabel').hidden,false);assert.equal(a.elements.get('playback').hidden,true);
  await a.click('drop');assert.equal(count(a,'verify'),1);assert.equal(count(a,'try'),0);await a.finish();
  await a.click('drop');await a.finish();assert.equal(count(a,'verify'),1);
  await a.click('nudgeRight');assert.equal(a.elements.get('ghostLabel').hidden,true);await a.click('drop');assert.equal(count(a,'try'),1);assert.equal(count(a,'verify'),1);
});

test('changing fields while a hint is pending cannot apply the stale suggestion',async()=>{
  const a=await boot({holdPlan:true});void a.elements.get('hint').click();await a.settle();assert.equal(a.elements.get('drop').disabled,true);
  await a.elements.get('fields').children[1].click();a.release();await a.settle();assert.match(a.elements.get('fieldName').textContent,/FIELD 02/);assert.equal(a.elements.get('moveValue').textContent,'0 cm');assert.equal(a.elements.get('ghostLabel').hidden,true);assert.equal(a.elements.get('drop').disabled,false);
});

test('invalid move displays an actionable error without inventing a replay',async()=>{
  const a=await boot({failTry:true});await a.move(100);await a.click('drop');assert.equal(a.elements.get('playback').hidden,true);assert.equal(a.elements.get('error').hidden,false);assert.match(a.elements.get('error').textContent,/overlaps/);assert.equal(a.elements.get('drop').disabled,false);assert.equal(a.elements.get('callValue').textContent,'1');
});

test('playing without a model remains possible; editor validates without a hidden simulation',async()=>{
  const a=await boot({model:false});assert.equal(a.elements.get('hint').disabled,true);await a.click('drop');assert.equal(count(a,'simulate'),1);
  await a.click('editField');assert.equal(a.elements.get('editorDialog').open,true);a.elements.get('ballX').value='1.5';
  await a.elements.get('editorForm').fire('submit');await a.settle();assert.equal(count(a,'validate'),1);assert.equal(count(a,'simulate'),1);assert.equal(a.elements.get('editorDialog').open,false);assert.equal(a.elements.get('playback').hidden,true);
});

test('bumper callout placement avoids the ball and target on the narrow fixture',async()=>{
  const a=await boot({width:390,height:844}),control=a.elements.get('bumperControl');
  const rect={x:parseFloat(control.style.left),y:parseFloat(control.style.top),w:190,h:76};
  const objects=vm.runInContext('[scene.ball, {position:goal.region.center,radius:goal.region.radius}].map(o=>({x:xy(o.position)[0]*camera.pixel,y:xy(o.position)[1]*camera.pixel,r:o.radius*camera.scale*camera.pixel}))',a.context);
  for(const o of objects){const dx=Math.max(rect.x-o.x,0,o.x-rect.x-rect.w),dy=Math.max(rect.y-o.y,0,o.y-rect.y-rect.h);assert.ok(Math.hypot(dx,dy)>o.r,'callout must not cover an important object');}
});


test('measured example loads exact geometry and move without borrowing a verification',async()=>{
  const a=await boot({contactExample:true});assert.equal(a.elements.get('contactResults').hidden,false);
  await a.click('drop');await a.finish();await a.click('openNotebook');await a.click('tryTolerance');
  assert.equal(a.elements.get('notebookDialog').open,false);assert.equal(a.elements.get('playback').hidden,true);
  assert.equal(a.elements.get('callValue').textContent,'0');assert.equal(count(a,'try'),0);assert.match(heading(a),/room for error/);
  await a.click('drop');const trial=a.calls.find(c=>c.name==='try');assert.equal(trial.body.scene.id,'measured');assert.equal(trial.body.edit_m,.061875);
  assert.equal(count(a,'verify'),0);assert.equal(a.elements.get('callValue').textContent,'2');await a.finish();
  await a.click('nudgeLeft');assert.equal(a.elements.get('playback').hidden,true);assert.doesNotMatch(heading(a),/room for error/);
});
