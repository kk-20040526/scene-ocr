const app = getApp();

Page({
  data: {
    serverBase: '',
    annotated: '',
    results: [],
    count: 0,
    loading: false,
    uploading: false
  },

  onLoad() {
    this.setData({
      serverBase: app.globalData.serverBase || 'http://127.0.0.1:8000'
    });
  },

  onServerInput(e) {
    this.setData({ serverBase: e.detail.value });
  },

  saveServer() {
    app.globalData.serverBase = this.data.serverBase;
    wx.setStorageSync('serverBase', this.data.serverBase);
    wx.showToast({ title: '已保存', icon: 'success' });
  },

  chooseImage() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      camera: 'back',
      success: (res) => {
        const path = res.tempFiles[0].tempFilePath;
        this.setData({ annotated: '', results: [], count: 0 });
        this.upload(path);
      }
    });
  },

  upload(filePath) {
    if (!this.data.serverBase) {
      wx.showToast({ title: '请先填写服务器地址', icon: 'none' });
      return;
    }
    this.setData({ uploading: true, loading: true });
    wx.uploadFile({
      url: this.data.serverBase.replace(/\/$/, '') + '/ocr',
      filePath: filePath,
      name: 'file',
      header: { 'content-type': 'multipart/form-data' },
      success: (res) => {
        if (res.statusCode !== 200) {
          wx.showToast({ title: '服务返回错误 ' + res.statusCode, icon: 'none' });
          return;
        }
        let data;
        try {
          data = JSON.parse(res.data);
        } catch (e) {
          wx.showToast({ title: '结果解析失败', icon: 'none' });
          return;
        }
        if (data.error) {
          wx.showToast({ title: data.error, icon: 'none' });
          return;
        }
        const results = data.results || [];
        this.setData({
          annotated: 'data:image/jpeg;base64,' + data.annotated_image,
          results: results,
          count: data.count || results.length
        });
      },
      fail: () => {
        wx.showModal({
          title: '连接失败',
          content: '请确认后端已启动、手机与电脑同一局域网，且服务器地址填写正确。',
          showCancel: false
        });
      },
      complete: () => {
        this.setData({ uploading: false, loading: false });
      }
    });
  },

  previewImage() {
    if (this.data.annotated) {
      wx.previewImage({ urls: [this.data.annotated] });
    }
  }
});
