// hsv.js — đếm hoa HSV/Otsu (thí nghiệm cũ) sau một công tắc, TẮT mặc định.
const DENS=[{id:0,name:'0 · Không hoa',color:'#64748b',lo:0,hi:2},{id:1,name:'1 · Ít hoa',color:'#22c55e',lo:2,hi:10},{id:2,name:'2 · Nhiều hoa',color:'#f59e0b',lo:10,hi:30},{id:3,name:'3 · Rất nhiều hoa',color:'#ef4444',lo:30,hi:1e9}];
function loadClasses(){try{const s=JSON.parse(localStorage.getItem('cofseg_density'));if(Array.isArray(s)&&s.length===4)return DENS.map((d,i)=>({id:d.id,name:d.name,color:(s[i]&&s[i].color)||d.color,lo:d.lo,hi:d.hi}));}catch(e){}return DENS.map(c=>({id:c.id,name:c.name,color:c.color,lo:c.lo,hi:c.hi}));}
function saveClasses(){localStorage.setItem('cofseg_density',JSON.stringify(classes));}
let classes=loadClasses();
let clsMax=3;
let activeCls=0;
let satMax=50,valMin=180;
// Phương pháp đếm hoa đang HIỂN THỊ và tham số Otsu. Cả hai kết quả luôn được
// server tính và lưu; `fmethod` chỉ quyết định con số nào lên màn hình và vào
// trường đầu của polygon khi lưu (label_method ghi lại điều đó).
let fmethod='hsv',otsuP={channel:'white',s_max:50,class_s_max:0,sep_min:0,area_min:0,area_max:0,elong_max:0,clip_rule:false};
let showMask=false,maskImg=null,maskTimer=null,maskToken=0,maskUrl=null;
// Công tắc của cả phần đếm hoa HSV/Otsu, TẮT mặc định. Tắt thì không gọi
// /api/flower*, không hiện mức/%, tán vẽ một màu, và lúc lưu server chỉ ghi
// hình polygon (cờ `flower` trong payload). Bật lại là y như trước.
let flowerOn=false;
const CROWN_COLOR='#60a5fa';
function flowerParams(){return Object.assign({sat_max:satMax,val_min:valMin,method:fmethod},otsuP);}
function loadFlowerPrefs(){try{const s=JSON.parse(localStorage.getItem('cofseg_flower'));if(!s)return;
 if(s.method==='hsv'||s.method==='otsu')fmethod=s.method;if(s.otsu)Object.assign(otsuP,s.otsu);showMask=!!s.showMask;flowerOn=!!s.on;}catch(e){}}
function saveFlowerPrefs(){try{localStorage.setItem('cofseg_flower',JSON.stringify({method:fmethod,otsu:otsuP,showMask:showMask,on:flowerOn}));}catch(e){}}
function methodName(){return fmethod==='hsv'?'HSV':'Otsu + blob';}
function pctTxt(v){return v==null?'—':v.toFixed(1)+'%';}
function densOf(pct){for(let i=classes.length-1;i>=0;i--){if(pct>=classes[i].lo)return classes[i].id;}return classes[0].id;}
function thrText(c){if(c.hi>=1e9)return '≥ '+c.lo+'%';if(c.lo<=0)return '< '+c.hi+'%';return c.lo+'–'+c.hi+'%';}
function clsById(id){return classes.find(c=>c.id===id);}
function clsColor(id){const c=clsById(id);return c?c.color:'#f97316';}
function renderClasses(){const box=$('classes');box.innerHTML='';
 classes.forEach(c=>{const row=document.createElement('div');row.className='cls'+(c.id===activeCls?' active':'');row.title='Chọn để gán thủ công mức "'+c.name+'" khi vẽ';
  const sw=document.createElement('input');sw.type='color';sw.value=c.color;sw.className='sw';
  sw.onclick=e=>e.stopPropagation();sw.oninput=()=>{c.color=sw.value;saveClasses();draw();renderRegions();};
  const nm=document.createElement('span');nm.className='cnm';nm.textContent=c.name;
  const thr=document.createElement('span');thr.className='cthr';thr.textContent=thrText(c);
  // Hiện thẳng số 0; bản cũ dùng `||''` nên ô trống trông như "chưa tính".
  const cnt=document.createElement('span');cnt.className='ccnt';cnt.textContent=inst.filter(r=>r.cls===c.id).length;
  row.onclick=()=>{activeCls=c.id;renderClasses();};
  row.appendChild(sw);row.appendChild(nm);row.appendChild(thr);row.appendChild(cnt);box.appendChild(row);});
 $('stCls').textContent=classes.length;}
// force = người dùng chủ ý bấm ⚡ -> bỏ cờ tay và phân loại lại. Các lần gọi
// ngầm (tạo vùng mới, nắn viền) luôn cập nhật % nhưng KHÔNG động vào nhãn tay.
// Giữ THAM CHIẾU vùng qua await, không giữ chỉ số: xoá một vùng trong lúc server
// đang đếm thì chỉ số trượt, kết quả gán nhầm sang vùng kế. Vùng không còn
// trong inst (đã xoá, hoặc Ctrl+Z thay cả mảng) thì bỏ qua kết quả của nó.
async function computeFlower(idxs,force){if(!flowerOn||!current)return;const targets=idxs.map(i=>inst[i]).filter(Boolean);if(!targets.length)return;
 const who=current,polys=targets.map(r=>r.poly);
 try{const r=await(await fetch('/api/flower',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(Object.assign({name:who,polygons:polys},flowerParams()))})).json();
  if(current!==who)return;
  if(r.error){alert('Không đếm được hoa: '+r.error);return;}
  (r.ratios||[]).forEach((rt,k)=>{const t=targets[k];if(!t||inst.indexOf(t)<0)return;
   t.pcts={hsv:(r.hsv&&r.hsv[k]!=null)?r.hsv[k]:null,otsu:(r.otsu&&r.otsu[k])?r.otsu[k].ratio:null};
   t.od=(r.otsu&&r.otsu[k])||null;
   t.pct=rt;
   if(force)t.manual=false;
   if(!t.manual)t.cls=densOf(rt);});
  persist();renderRegions();renderClasses();draw();}catch(e){alert('Không đếm được hoa (đã khởi động lại server chưa?): '+e);}}
function autoAll(){if(!flowerOn)return;if(!inst.length){alert('Chưa có vùng nào để phân loại.');return;}pushHist();computeFlower(inst.map((_,i)=>i),true);}
function syncFlowerUI(){$('fmethod').value=fmethod;$('hsvParams').hidden=(fmethod!=='hsv');$('otsuParams').hidden=(fmethod!=='otsu');
 $('ochan').value=otsuP.channel;$('sMax').value=otsuP.s_max;$('classSMax').value=otsuP.class_s_max;$('sMaxRow').hidden=(otsuP.channel!=='white');
 $('sepMin').value=otsuP.sep_min;$('areaMin').value=otsuP.area_min;$('areaMax').value=otsuP.area_max;$('elongMax').value=otsuP.elong_max;$('clipRule').checked=!!otsuP.clip_rule;$('showMask').checked=showMask;}
// Lớp phủ pixel hoa: PNG trong suốt cỡ cả ảnh do server vẽ theo phương pháp đang
// chọn. Làm mới có trễ 400 ms để kéo điểm liên tục không dội server; token và
// tên ảnh chặn kết quả cũ đè lên ảnh mới.
function scheduleMask(){clearTimeout(maskTimer);if(!flowerOn||!showMask){maskImg=null;draw();return;}maskTimer=setTimeout(fetchMask,400);}
async function fetchMask(){if(!flowerOn||!showMask||!current||!ready||!inst.length){maskImg=null;draw();return;}
 const who=current,tok=++maskToken;
 try{const r=await fetch('/api/flower_mask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(Object.assign({name:who,polygons:inst.map(x=>x.poly)},flowerParams()))});
  if(!r.ok)throw new Error('HTTP '+r.status);const blob=await r.blob();
  if(tok!==maskToken||current!==who)return;
  const im=new Image();im.onload=()=>{if(tok!==maskToken||current!==who)return;if(maskUrl)URL.revokeObjectURL(maskUrl);maskUrl=im.src;maskImg=im;draw();};
  im.src=URL.createObjectURL(blob);}
 catch(e){maskImg=null;draw();}}
// Bật/tắt chỉ đổi cách HIỂN THỊ và việc lưu về sau, không tự lưu gì: bật lên
// thì vùng nào chưa có % vẫn để "—" cho tới khi bấm ⚡.
function syncFlowerOn(){const b=$('flowerOn');b.classList.toggle('on',flowerOn);
 b.title=flowerOn?'Đang bật: đếm pixel hoa HSV/Otsu cho từng tán và lưu vào file nhãn. Bấm để tắt.'
  :'Đang tắt: chỉ lưu hình polygon, không đọc ảnh để đếm hoa. Bấm để bật (thí nghiệm cũ, mật độ hoa sẽ tính từ mask hoa).';
 $('flowerBox').hidden=!flowerOn;$('stClsBox').hidden=!flowerOn;
 document.querySelector('#setModal .hsvbox').hidden=!flowerOn;}
function pickCls(id){return clsById(id)?id:activeCls;}
function mergeCats(cats){if(!cats)return;let ch=false;cats.forEach(c=>{if(c&&c.id!=null&&!clsById(c.id)){classes.push({id:c.id,name:c.name||('Lớp '+c.id),color:c.color||'#888'});if(c.id>clsMax)clsMax=c.id;ch=true;}});if(ch){saveClasses();renderClasses();renderStats();}}
wsRegister({id:'hsv',
 resetImage(){maskImg=null;maskToken++;clearTimeout(maskTimer);},
 onImage(){renderClasses();scheduleMask();},
 draw(){if(wsActive!=='flower'&&flowerOn&&showMask&&maskImg)ctx.drawImage(maskImg,ox,oy,img.width*scale,img.height*scale);}});
