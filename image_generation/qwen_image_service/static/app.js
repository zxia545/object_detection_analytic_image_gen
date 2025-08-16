// 通用函数
function qs(sel) { return document.querySelector(sel);}
function qsa(sel){ return Array.from(document.querySelectorAll(sel));}

const API = "" // 同源/default

// 选项卡切换
qsa('nav button').forEach(btn=>{
    btn.onclick = ()=>{
        qsa('nav button').forEach(x=>x.classList.remove('active'));
        btn.classList.add('active');
        ["gen", "queue", "history"].forEach(type=>{
            qs('#'+type+"-section").classList.toggle('hidden', btn.id!==type+"-tab");
        });
        if(btn.id==='queue-tab') updateQueue();
        if(btn.id==='history-tab') loadHistory(0);
    }
});

// ------------------------ 1. 生成图片 ------------------------
const genForm = qs('#gen-form');
const submitTip = qs('#submit-tip');
const progressDiv = qs('#gen-progress'), progressTxt = qs('#gen-status-text');
const resultDiv = qs('#gen-result');

let polling = null;
genForm.onsubmit = async function(e){
    e.preventDefault();
    resultDiv.classList.add('hidden');
    progressDiv.classList.remove('hidden');
    progressTxt.innerText = "图片生成中，请稍等...";
    submitTip.innerText = '';

    const fd = new FormData(genForm);
    const body = {};
    [ ...fd.entries() ].forEach(([k,v])=>{ body[k]=v }); 
    if(!body['negative_prompt'])body['negative_prompt']="";

    try{
        const resp = await fetch(API+"/generate", {
            method:"POST",
            headers: { 'Content-Type':'application/json' },
            body: JSON.stringify(body)
        });
        if(!resp.ok) throw new Error("提交失败");
        const data = await resp.json();
        const task_id = data.task_id;
        // 轮询查询状态
        pollTask(task_id);
    }catch(e){
        progressDiv.classList.add('hidden');
        showResult({error: '提交失败：'+e});
    }
};

function pollTask(task_id){
    if(polling) {
        clearInterval(polling);
        polling = null;
    }
    let timer = null;
    async function poll(){
        try{
            const resp = await fetch(API+`/status/${task_id}`);
            if(!resp.ok){
                showResult({error:"任务查询失败..."});
                clearInterval(timer);
                polling = null;
                return;
            }
            const data = await resp.json();
            if(data.status === 'failed'){
                showResult({error:"任务失败: "+(data.detail||"未知错误")});
                clearInterval(timer);
                polling = null;
            }else if(data.status==='done'){
                showResult({url:data.result_url, task_id});
                clearInterval(timer);
                polling = null;
            }else{
                // 继续轮询
                progressTxt.innerText = "图片生成中，请耐心等待...(状态："+data.status+")";
            }
        }catch(e){
            showResult({error:"状态轮询异常."});
            clearInterval(timer);
            polling = null;
        }
    }
    poll();
    timer = setInterval(poll, 2100);
    polling = timer;
}

function showResult({url,error,task_id}){
    progressDiv.classList.add('hidden');
    resultDiv.innerHTML = '';
    if(error){
        resultDiv.innerHTML = `<div class="genfail">❌ 生成失败: ${error}</div>`;
    }else if(url){
        const img_url = url.startsWith("/")?url:("/"+url);
        resultDiv.innerHTML =
            `<img src="${img_url}" alt="生成结果">
             <a href="${img_url}" download="result.png"><button class="download-btn">下载图片</button></a>
             <div style="margin-top:6px;font-size:.97em;color:#888;">任务ID: ${task_id}</div>
            `;
    }
    resultDiv.classList.remove('hidden');
}

// ------------------------ 2. 队列管理 ------------------------
async function updateQueue(){
    const table = qs('#queue-table tbody');
    table.innerHTML = "<tr><td colspan=3>加载中...</td></tr>";
    try{
        const resp = await fetch(API+"/queue");
        if(!resp.ok) throw new Error("队列查询失败");
        const list = await resp.json();
        if(!list.length){
            table.innerHTML = "<tr><td colspan=3>当前队列为空</td></tr>";
            return;
        }
        table.innerHTML = "";
        for(const t of list){
            table.innerHTML += `
            <tr>
                <td>${t.task_id}</td>
                <td>${statusText(t.status)}</td>
                <td>
                    <button onclick="checkQueueTask('${t.task_id}')">查看状态</button>
                </td>
            </tr>`;
        }
    }catch(e){
        table.innerHTML = `<tr><td colspan=3>加载失败:${e}</td></tr>`;
    }
}
qs('#refresh-queue').onclick = updateQueue;

window.checkQueueTask = async function(task_id){
    // 切换到生成栏目并直接查询任务
    genForm.reset();
    progressDiv.classList.remove('hidden');
    progressTxt.innerText = "查询任务状态中...";
    resultDiv.classList.add('hidden');
    qsa('nav button').forEach(x=>x.classList.remove('active'));
    qs('#gen-tab').classList.add('active');
    ["gen", "queue", "history"].forEach(type=>{
        qs('#'+type+"-section").classList.toggle('hidden', type!=="gen");
    });
    pollTask(task_id);
};

// ------------------------ 3. 历史记录 ------------------------
let historySkip = 0, historyLimit = 10, historyTotal = 0;
async function loadHistory(skip){
    const table=qs('#history-table tbody');
    table.innerHTML = "<tr><td colspan=4>加载中...</td></tr>";
    try{
        const resp = await fetch(API+`/history?skip=${skip}&limit=${historyLimit}`);
        if(!resp.ok) throw new Error("历史加载失败");
        const data = await resp.json();
        historyTotal = data.total;
        const items = data.items;
        if(!items.length){
            table.innerHTML = "<tr><td colspan=4>历史记录为空</td></tr>";
            qs('#history-page-info').innerText="无记录";
            return;
        }
        table.innerHTML = "";
        for(const t of items){
            let time = "--";
            if(t.ctime){
                time = new Date(t.ctime*1000).toLocaleString();
            }
            let pic = t.status=="done" && t.result_url ? `<a href="${t.result_url}" target="_blank"><img src="${t.result_url}"></a>` : `<span class='fail'>${t.detail||""}</span>`;
            table.innerHTML += 
                `<tr>
                    <td style="font-size:.97em">${t.task_id}</td>
                    <td>${statusText(t.status)}</td>
                    <td>${time}</td>
                    <td>${pic}</td>
                </tr>`;
        }
        qs('#history-page-info').innerText = `第 ${Math.floor(skip/historyLimit)+1} 页 / 共 ${Math.ceil(historyTotal/historyLimit)} 页`;
    }catch(e){
        table.innerHTML = `<tr><td colspan=4>加载失败:${e}</td></tr>`;
        qs('#history-page-info').innerText = "";
    }
}
qs('#history-prev').onclick = ()=>{
    if(historySkip>=historyLimit){
        historySkip -= historyLimit;
        loadHistory(historySkip);
    }
};
qs('#history-next').onclick = ()=>{
    if(historySkip+historyLimit < historyTotal){
        historySkip += historyLimit;
        loadHistory(historySkip);
    }
};

function getCTime(f){ return 0; } // 占位，实际ctime仅后端用

function statusText(s){
    if(s=="pending") return "排队中";
    if(s=="running") return "生成中";
    if(s=="done")    return "已完成";
    if(s=="loading") return "加载模型中";
    if(s=="failed")  return "失败";
    return s;
}

// 默认首页加载