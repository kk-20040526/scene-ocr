App({
  globalData: {
    // 真机调试：手机与电脑同一局域网，把这里改成电脑的局域网 IP（如 http://192.168.1.10:8000）
    // 开发者工具（本机）可用 http://127.0.0.1:8000
    serverBase: 'http://127.0.0.1:8000'
  },
  onLaunch() {
    const saved = wx.getStorageSync('serverBase');
    if (saved) {
      this.globalData.serverBase = saved;
    }
  }
});
