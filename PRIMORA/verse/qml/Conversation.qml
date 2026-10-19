import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// V1 conversation area: message history + text input
Item {
    id: conversation
    objectName: "conversationView"

    // Message model for the ListView
    ListModel {
        id: messageModel
    }

    // Connect to verseApp signals using modern Connections syntax
    // verseApp is a context property set on the root context
    Connections {
        id: verseAppConnections
        target: verseApp
        function onMessageReceived(role, content, timestamp) {
            messageModel.append({
                role: role,
                content: content,
                timestamp: timestamp
            })
            // Auto-scroll to bottom
            listView.positionViewAtEnd()
        }
        function onMessageUpdated(index, content, timestamp) {
            // Update a single assistant bubble in place (streaming).
            if (index >= 0 && index < messageModel.count) {
                messageModel.set(index, {
                    content: content,
                    timestamp: timestamp
                })
                listView.positionViewAtEnd()
            }
        }
        // Python-initiated clears (clear_conversation) reach the model
        // through the existing signal, so both directions stay in sync.
        function onConversationCleared() {
            messageModel.clear()
        }
    }

    // Compact centered stream: the conversation is an overlay inside the
    // Universe, so it stays narrow and low rather than a full-width panel.
    Item {
        anchors.fill: parent

        ColumnLayout {
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            anchors.horizontalCenter: parent.horizontalCenter
            width: Math.min(parent.width * 0.62, 620)

        // Conversation controls: Clear is user-visible, keyboard-focusable,
        // and routes through the same clearConversation() path the QML side
        // owns (model first, then the Python slot and its marker row).
        RowLayout {
            Layout.fillWidth: true
            Layout.bottomMargin: 2

            Item { Layout.fillWidth: true }

            Button {
                id: clearButton
                objectName: "clearConversationButton"
                text: "Clear Conversation"
                Layout.minimumHeight: 28
                background: Rectangle {
                    color: clearButton.hovered ? "#1a2636" : "#121a26"
                    radius: 8
                    border.color: clearButton.activeFocus ? "#33506e" : "#232f45"
                    border.width: 1
                }
                font.family: "Segoe UI"
                font.pixelSize: 11
                palette.buttonText: "#8f9fb4"
                onClicked: conversation.clearConversation()
            }
        }

        // Message history
        ListView {
            id: listView
            objectName: "verseMessageList"
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 200
            model: messageModel
            clip: true
            spacing: 6

            delegate: Item {
                width: listView.width
                height: msgColumn.implicitHeight + 12

                Rectangle {
                    width: Math.min(parent.width * 0.58, 420)
                    height: msgColumn.implicitHeight + 10
                    anchors.top: parent.top
                    anchors.topMargin: 1
                    anchors.left: role === "user" ? undefined : parent.left
                    anchors.right: role === "user" ? parent.right : undefined
                    anchors.leftMargin: 4
                    anchors.rightMargin: 4
                    color: role === "user" ? "#182a3d" : "#141a24"
                    opacity: role === "user" ? 0.92 : 0.85
                    radius: 10
                    border.color: role === "user" ? "#33506e" : "#2a3547"
                    border.width: 1

                    Column {
                        id: msgColumn
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.leftMargin: 10
                        anchors.rightMargin: 10
                        anchors.topMargin: 5
                        spacing: 2

                        Text {
                            id: roleLabel
                            text: role === "user" ? "Amir" : "Aurora"
                            color: role === "user" ? "#7ba7d4" : "#8f9fb4"
                            font.family: "Segoe UI"
                            font.pixelSize: 11
                            font.letterSpacing: 1.1
                        }

                        Text {
                            id: messageText
                            text: content
                            color: "#c3cfdd"
                            font.family: "Segoe UI"
                            font.pixelSize: 13
                            wrapMode: Text.WordWrap
                            width: parent.width
                            lineHeight: 1.35
                        }
                    }
                }
            }
        }

        // Input area
        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            TextArea {
                id: inputField
                objectName: "verseInputField"
                Layout.fillWidth: true
                Layout.minimumHeight: 40
                Layout.maximumHeight: 110
                placeholderText: "Speak to Aurora…  (Enter sends · Shift+Enter newline)"
                placeholderTextColor: "#5a6b84"
                wrapMode: TextArea.Wrap
                font.family: "Segoe UI"
                font.pixelSize: 13
                color: "#cfdae8"
                topPadding: 10
                bottomPadding: 10
                leftPadding: 12
                rightPadding: 12
                background: Rectangle {
                    color: "#0a111b"
                    radius: 10
                    border.color: "#232f45"
                    border.width: 1
                }
                selectionColor: "#2c4a68"
                selectedTextColor: "#ffffff"

                Keys.onPressed: function(event) {
                    var isSendKey = (event.key === Qt.Key_Return
                        || event.key === Qt.Key_Enter)
                    var shiftHeld = (event.modifiers & Qt.ShiftModifier) !== 0
                    if (isSendKey && !shiftHeld) {
                        event.accepted = true
                        sendMessage()
                    }
                }
            }

            Button {
                id: sendButton
                Layout.preferredWidth: 64
                Layout.minimumHeight: 40
                text: "Send"
                enabled: inputField.text.trim().length > 0 && !verseApp.isResponding
                background: Rectangle {
                    color: enabled ? "#1f3a52" : "#121a26"
                    radius: 10
                    border.color: enabled ? "#33506e" : "#1d2839"
                    border.width: 1
                }
                font.family: "Segoe UI"
                font.pixelSize: 13
                palette.buttonText: enabled ? "#cfe3f5" : "#5a6b84"
                onClicked: sendMessage()
            }
        }
        } // centered stream wrapper
    }

    function sendMessage() {
        if (verseApp && verseApp.isResponding) return
        var text = inputField.text.trim()
        if (text.length === 0) return
        if (verseApp) {
            verseApp.send_message(text)
        }
        inputField.text = ""
    }

    function clearConversation() {
        messageModel.clear()
        if (verseApp) {
            verseApp.clear_conversation()
        }
    }
}